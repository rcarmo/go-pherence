package nemotrondiarization

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const (
	diarizationHeads     = 8
	diarizationHeadWidth = 64
)

// Layer0Attention owns the first released audio attention block's weights.
// This bounded offline path uses a full bidirectional window, with no masks,
// cache, later MLP/encoder layers, upsampler or speaker classification.
type Layer0Attention struct {
	qkv       *Layer0QKV
	outWeight []float32
	outBias   []float32
	outPacked []float32
}

func LoadLayer0Attention(file *safetensors.File) (*Layer0Attention, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron diarization checkpoint")
	}
	qkv, err := LoadLayer0QKV(file)
	if err != nil {
		return nil, err
	}
	weight, shape, err := file.GetFloat32("model.audio_tower.layers.0.self_attn.o_proj.weight")
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 || shape[0] != projectedWidth || shape[1] != projectedWidth || len(weight) != projectedWidth*projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization attention output weight shape %v", shape)
	}
	bias, shape, err := file.GetFloat32("model.audio_tower.layers.0.self_attn.o_proj.bias")
	if err != nil {
		return nil, err
	}
	if len(shape) != 1 || shape[0] != projectedWidth || len(bias) != projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization attention output bias shape %v", shape)
	}
	for _, tensor := range [][]float32{weight, bias} {
		for _, value := range tensor {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite Nemotron diarization attention weight")
			}
		}
	}
	m := &Layer0Attention{qkv: qkv, outWeight: weight, outBias: bias}
	if simd.HasSgemmAsm {
		m.outPacked, err = simd.PackSgemmNTWeights(weight, projectedWidth, projectedWidth, projectedWidth)
		if err != nil {
			return nil, err
		}
	}
	return m, nil
}

// ForwardOffline returns independent owned attention and residual outputs.
// Input is a complete, unmasked [rows,512] stacking window with position zero.
// Attention is bidirectional within exactly these rows; slicing a longer
// recording changes the attention context and hence its numerical result.
func (m *Layer0Attention) ForwardOffline(input []float32, rows int) (attention, residual []float32, err error) {
	return m.forwardOffline(input, rows, simd.HasSgemmAsm)
}

func (m *Layer0Attention) forwardOfflineScalar(input []float32, rows int) (attention, residual []float32, err error) {
	return m.forwardOffline(input, rows, false)
}

func (m *Layer0Attention) forwardOffline(input []float32, rows int, vector bool) (attention, residual []float32, err error) {
	if m == nil || m.qkv == nil || len(m.outWeight) != projectedWidth*projectedWidth || len(m.outBias) != projectedWidth {
		return nil, nil, fmt.Errorf("invalid Nemotron diarization attention model")
	}
	if rows < 1 || rows > maxPreparedDiarizationRows || len(input) != rows*projectedWidth {
		return nil, nil, fmt.Errorf("invalid Nemotron diarization attention window")
	}
	inputNormal, q, k, v, err := m.qkv.projectWithInputNormal(input, rows)
	if err != nil {
		return nil, nil, err
	}
	// Transformers uses the two-half rotation: [-second_half, first_half].
	var inv [diarizationHeadWidth / 2]float32
	for i := range inv {
		inv[i] = float32(1 / math.Pow(10000, float64(2*i)/diarizationHeadWidth))
	}
	for row := 0; row < rows; row++ {
		// Each head uses the same position and frequencies. Keep the F32
		// angle rounding while evaluating sine/cosine once per position.
		var cosine, sine [diarizationHeadWidth / 2]float32
		for dim := range inv {
			angle := float64(float32(row) * inv[dim])
			cosine[dim], sine[dim] = float32(math.Cos(angle)), float32(math.Sin(angle))
		}
		for head := 0; head < diarizationHeads; head++ {
			base := row*projectedWidth + head*diarizationHeadWidth
			for dim := range inv {
				for _, values := range [][]float32{q, k} {
					a, b := values[base+dim], values[base+dim+diarizationHeadWidth/2]
					values[base+dim] = a*cosine[dim] - b*sine[dim]
					values[base+dim+diarizationHeadWidth/2] = b*cosine[dim] + a*sine[dim]
				}
			}
		}
	}
	// Each Q head is dead after its scores are computed. The tiled path
	// copies Q before scoring; the scalar path finishes a row's scores
	// before writing its mixed values. Reuse the owned Q buffer.
	mixed := q
	const scaling = float32(1.0 / 8.0) // head width 64, inverse square root
	if vector && rows >= 16 {
		// Pack each head into contiguous rows for checked Q*K^T and P*V.
		// The scalar path below avoids this preparation on small windows and
		// machines without an SGEMM kernel.
		qHead, kHead, vHead := make([]float32, rows*diarizationHeadWidth), make([]float32, rows*diarizationHeadWidth), make([]float32, rows*diarizationHeadWidth)
		scores := make([]float32, rows*rows)
		for head := 0; head < diarizationHeads; head++ {
			for row := 0; row < rows; row++ {
				from := row*projectedWidth + head*diarizationHeadWidth
				to := row * diarizationHeadWidth
				copy(qHead[to:to+diarizationHeadWidth], q[from:from+diarizationHeadWidth])
				copy(kHead[to:to+diarizationHeadWidth], k[from:from+diarizationHeadWidth])
				copy(vHead[to:to+diarizationHeadWidth], v[from:from+diarizationHeadWidth])
			}
			clear(scores)
			if !simd.DenseNTTo(scores, qHead, kHead, rows, rows, diarizationHeadWidth, scaling, diarizationHeadWidth, diarizationHeadWidth, rows) {
				return nil, nil, fmt.Errorf("Nemotron diarization attention score shape rejected")
			}
			for row := 0; row < rows; row++ {
				if !simd.SoftmaxSIMDInPlace(scores[row*rows : (row+1)*rows]) {
					return nil, nil, fmt.Errorf("Nemotron diarization attention softmax failed")
				}
			}
			clear(qHead)
			if !simd.SgemmNNTo(qHead, scores, vHead, rows, diarizationHeadWidth, rows, 1, rows, diarizationHeadWidth, diarizationHeadWidth) {
				return nil, nil, fmt.Errorf("Nemotron diarization attention value shape rejected")
			}
			for row := 0; row < rows; row++ {
				copy(mixed[row*projectedWidth+head*diarizationHeadWidth:row*projectedWidth+(head+1)*diarizationHeadWidth], qHead[row*diarizationHeadWidth:(row+1)*diarizationHeadWidth])
			}
		}
	} else {
		scores := make([]float32, rows)
		for head := 0; head < diarizationHeads; head++ {
			base := head * diarizationHeadWidth
			for row := 0; row < rows; row++ {
				for source := 0; source < rows; source++ {
					var sum float32
					for dim := 0; dim < diarizationHeadWidth; dim++ {
						sum += q[row*projectedWidth+base+dim] * k[source*projectedWidth+base+dim]
					}
					scores[source] = sum * scaling
				}
				if !simd.SoftmaxInPlace(scores) {
					return nil, nil, fmt.Errorf("Nemotron diarization attention softmax failed")
				}
				for dim := 0; dim < diarizationHeadWidth; dim++ {
					var sum float32
					for source := 0; source < rows; source++ {
						sum += scores[source] * v[source*projectedWidth+base+dim]
					}
					mixed[row*projectedWidth+base+dim] = sum
				}
			}
		}
	}
	attention = make([]float32, len(input))
	if !diarizationDenseMLP(attention, mixed, m.outWeight, m.outPacked, rows, projectedWidth, projectedWidth) {
		return nil, nil, fmt.Errorf("Nemotron diarization attention output rejected shape")
	}
	residual = make([]float32, len(input))
	for i, value := range attention {
		attention[i] = value + m.outBias[i%projectedWidth]
		residual[i] = inputNormal[i] + attention[i]
	}
	return attention, residual, nil
}
