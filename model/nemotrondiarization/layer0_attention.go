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
	return &Layer0Attention{qkv: qkv, outWeight: weight, outBias: bias}, nil
}

// ForwardOffline returns independent owned attention and residual outputs.
// Input is a complete, unmasked [rows,512] stacking window with position zero.
// Attention is bidirectional within exactly these rows; slicing a longer
// recording changes the attention context and hence its numerical result.
func (m *Layer0Attention) ForwardOffline(input []float32, rows int) (attention, residual []float32, err error) {
	if m == nil || m.qkv == nil || len(m.outWeight) != projectedWidth*projectedWidth || len(m.outBias) != projectedWidth {
		return nil, nil, fmt.Errorf("invalid Nemotron diarization attention model")
	}
	if rows < 1 || rows > 376 || len(input) != rows*projectedWidth {
		return nil, nil, fmt.Errorf("invalid Nemotron diarization attention window")
	}
	q, k, v, err := m.qkv.Project(input, rows)
	if err != nil {
		return nil, nil, err
	}
	inputNormal := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(inputNormal, input, rows, projectedWidth, m.qkv.inputGamma, m.qkv.inputBeta, 1e-5) {
		return nil, nil, fmt.Errorf("Nemotron diarization input normalisation rejected window")
	}
	// Transformers uses the two-half rotation: [-second_half, first_half].
	var inv [diarizationHeadWidth / 2]float32
	for i := range inv {
		inv[i] = float32(1 / math.Pow(10000, float64(2*i)/diarizationHeadWidth))
	}
	for row := 0; row < rows; row++ {
		for head := 0; head < diarizationHeads; head++ {
			base := row*projectedWidth + head*diarizationHeadWidth
			for dim := 0; dim < diarizationHeadWidth/2; dim++ {
				angle := float64(float32(row) * inv[dim])
				cosine, sine := float32(math.Cos(angle)), float32(math.Sin(angle))
				for _, values := range [][]float32{q, k} {
					a, b := values[base+dim], values[base+dim+diarizationHeadWidth/2]
					values[base+dim] = a*cosine - b*sine
					values[base+dim+diarizationHeadWidth/2] = b*cosine + a*sine
				}
			}
		}
	}
	mixed := make([]float32, rows*projectedWidth)
	scores := make([]float32, rows)
	const scaling = float32(1.0 / 8.0) // head width 64, inverse square root
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
	attention = make([]float32, len(input))
	if !simd.DenseNTTo(attention, mixed, m.outWeight, rows, projectedWidth, projectedWidth, 1, projectedWidth, projectedWidth, projectedWidth) {
		return nil, nil, fmt.Errorf("Nemotron diarization attention output rejected shape")
	}
	residual = make([]float32, len(input))
	for i, value := range attention {
		attention[i] = value + m.outBias[i%projectedWidth]
		residual[i] = inputNormal[i] + attention[i]
	}
	return attention, residual, nil
}
