package nemotrondiarization

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Layer1Attention owns the second audio layer's full-window attention weights.
// It does not implement the following MLP or any later encoder layers.
type Layer1Attention struct {
	qkv                *Layer1QKV
	outWeight, outBias []float32
}

func LoadLayer1Attention(file *safetensors.File) (*Layer1Attention, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron diarization checkpoint")
	}
	qkv, err := LoadLayer1QKV(file)
	if err != nil {
		return nil, err
	}
	prefix := "model.audio_tower.layers.1.self_attn.o_proj."
	weight, shape, err := file.GetFloat32(prefix + "weight")
	if err != nil {
		return nil, err
	}
	if len(shape) != 2 || shape[0] != projectedWidth || shape[1] != projectedWidth || len(weight) != projectedWidth*projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization layer-1 attention output weight shape %v", shape)
	}
	bias, shape, err := file.GetFloat32(prefix + "bias")
	if err != nil {
		return nil, err
	}
	if len(shape) != 1 || shape[0] != projectedWidth || len(bias) != projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization layer-1 attention output bias shape %v", shape)
	}
	for _, values := range [][]float32{weight, bias} {
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite Nemotron diarization layer-1 attention weight")
			}
		}
	}
	return &Layer1Attention{qkv: qkv, outWeight: weight, outBias: bias}, nil
}

// ForwardOffline returns owned attention and residual arrays from a complete
// unmasked [rows,512] layer-0 output. Positions start at zero; slicing an
// existing window changes bidirectional attention context.
func (m *Layer1Attention) ForwardOffline(input []float32, rows int) (attention, residual []float32, err error) {
	return m.forwardOffline(input, rows, simd.HasSgemmAsm)
}

func (m *Layer1Attention) forwardOfflineScalar(input []float32, rows int) (attention, residual []float32, err error) {
	return m.forwardOffline(input, rows, false)
}

func (m *Layer1Attention) forwardOffline(input []float32, rows int, vector bool) (attention, residual []float32, err error) {
	if m == nil || m.qkv == nil || len(m.outWeight) != projectedWidth*projectedWidth || len(m.outBias) != projectedWidth {
		return nil, nil, fmt.Errorf("invalid Nemotron diarization layer-1 attention model")
	}
	q, k, v, err := m.qkv.Project(input, rows)
	if err != nil {
		return nil, nil, err
	}
	var inv [diarizationHeadWidth / 2]float32
	for i := range inv {
		inv[i] = float32(1 / math.Pow(10000, float64(2*i)/diarizationHeadWidth))
	}
	for row := 0; row < rows; row++ {
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
	mixed := make([]float32, len(input))
	const scaling = float32(1.0 / 8.0)
	if vector && rows >= 16 {
		qHead, kHead, vHead := make([]float32, rows*diarizationHeadWidth), make([]float32, rows*diarizationHeadWidth), make([]float32, rows*diarizationHeadWidth)
		scores := make([]float32, rows*rows)
		product := make([]float32, rows*diarizationHeadWidth)
		for head := 0; head < diarizationHeads; head++ {
			for row := 0; row < rows; row++ {
				from := row*projectedWidth + head*diarizationHeadWidth
				to := row * diarizationHeadWidth
				copy(qHead[to:to+diarizationHeadWidth], q[from:from+diarizationHeadWidth])
				copy(kHead[to:to+diarizationHeadWidth], k[from:from+diarizationHeadWidth])
				copy(vHead[to:to+diarizationHeadWidth], v[from:from+diarizationHeadWidth])
			}
			clear(scores)
			if !simd.SgemmNTTo(scores, qHead, kHead, rows, rows, diarizationHeadWidth, scaling, diarizationHeadWidth, diarizationHeadWidth, rows) {
				return nil, nil, fmt.Errorf("Nemotron diarization layer-1 score shape rejected")
			}
			if !simd.SoftmaxRowsInPlace(scores, rows, rows) {
				return nil, nil, fmt.Errorf("Nemotron diarization layer-1 softmax failed")
			}
			clear(product)
			if !simd.SgemmNNTo(product, scores, vHead, rows, diarizationHeadWidth, rows, 1, rows, diarizationHeadWidth, diarizationHeadWidth) {
				return nil, nil, fmt.Errorf("Nemotron diarization layer-1 value shape rejected")
			}
			for row := 0; row < rows; row++ {
				copy(mixed[row*projectedWidth+head*diarizationHeadWidth:row*projectedWidth+(head+1)*diarizationHeadWidth], product[row*diarizationHeadWidth:(row+1)*diarizationHeadWidth])
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
					return nil, nil, fmt.Errorf("Nemotron diarization layer-1 softmax failed")
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
	if !simd.DenseNTTo(attention, mixed, m.outWeight, rows, projectedWidth, projectedWidth, 1, projectedWidth, projectedWidth, projectedWidth) {
		return nil, nil, fmt.Errorf("Nemotron diarization layer-1 output rejected shape")
	}
	residual = make([]float32, len(input))
	for i, value := range attention {
		attention[i] = value + m.outBias[i%projectedWidth]
		residual[i] = input[i] + attention[i]
		if math.IsNaN(float64(residual[i])) || math.IsInf(float64(residual[i]), 0) {
			return nil, nil, fmt.Errorf("non-finite Nemotron diarization layer-1 output")
		}
	}
	return attention, residual, nil
}
