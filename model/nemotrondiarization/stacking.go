// Package nemotrondiarization contains bounded native operators for the
// pinned NVIDIA Nemotron 3 Diarization checkpoint. It does not yet run the
// transformer, streaming speaker cache or classification head.
package nemotrondiarization

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const (
	melBins        = 128
	stackFrames    = 8
	stackWidth     = melBins * stackFrames
	projectedWidth = 512
	projectionName = "model.audio_tower.embedder.projection.weight"
)

// StackingProjection owns the F32 row-major [512,1024] released weight.
// The caller can close the safetensors file after loading it.
type StackingProjection struct{ weight []float32 }

func LoadStackingProjection(file *safetensors.File) (*StackingProjection, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron diarization checkpoint")
	}
	values, shape, err := file.GetFloat32(projectionName)
	if err != nil {
		return nil, fmt.Errorf("Nemotron diarization stack weight: %w", err)
	}
	if len(shape) != 2 || shape[0] != projectedWidth || shape[1] != stackWidth || len(values) != projectedWidth*stackWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization stack weight shape %v", shape)
	}
	for _, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization stack weight")
		}
	}
	return &StackingProjection{weight: values}, nil
}

// Project turns frame-major [frames,128] mel features into owned
// [ceil(frames/8),512] embeddings. The last group is zero-padded. It never
// mutates caller input or retains it; SIMD dispatch is checked by the backend.
func (p *StackingProjection) Project(features []float32, frames int) ([]float32, error) {
	return p.project(features, frames, true)
}

// ProjectScalar uses the same released weight and scalar reduction order.
// The independently generated PyTorch fixture supplies numerical acceptance.
func (p *StackingProjection) ProjectScalar(features []float32, frames int) ([]float32, error) {
	return p.project(features, frames, false)
}

func (p *StackingProjection) project(features []float32, frames int, vector bool) ([]float32, error) {
	if p == nil || len(p.weight) != projectedWidth*stackWidth {
		return nil, fmt.Errorf("invalid Nemotron diarization stacking projection")
	}
	if frames < 1 || frames > 3001 || len(features) != frames*melBins {
		return nil, fmt.Errorf("invalid Nemotron diarization feature shape")
	}
	for _, v := range features {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron diarization features")
		}
	}
	rows := (frames + stackFrames - 1) / stackFrames
	stacked := make([]float32, rows*stackWidth)
	copy(stacked, features)
	out := make([]float32, rows*projectedWidth)
	if vector && simd.HasSgemmAsm {
		if !simd.DenseNTTo(out, stacked, p.weight, rows, projectedWidth, stackWidth, 1, stackWidth, stackWidth, projectedWidth) {
			return nil, fmt.Errorf("Nemotron diarization projection rejected validated shape")
		}
	} else {
		for row := 0; row < rows; row++ {
			for channel := 0; channel < projectedWidth; channel++ {
				var sum float32
				for i, x := range stacked[row*stackWidth : (row+1)*stackWidth] {
					sum += x * p.weight[channel*stackWidth+i]
				}
				out[row*projectedWidth+channel] = sum
			}
		}
	}
	return out, nil
}
