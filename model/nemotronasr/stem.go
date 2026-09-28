// Package nemotronasr contains bounded native operators for the pinned
// Nemotron 3.5 ASR checkpoint. Full subsampling, RNNT and streaming state are
// not implemented yet.
package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const stemChannels = 256

// StemConv2D owns released stem weights in PyTorch [256,1,3,3] layout.
// Offline computes the first causal Conv2D only; it does not apply ReLU,
// valid-length masking, subsequent depthwise stages, or encoder projection.
type StemConv2D struct {
	Weight []float32
	Bias   []float32
}

// LoadStemConv2D copies the released F32 stem tensors. The caller can close
// the safetensors file after loading; no mmap view is retained.
func LoadStemConv2D(file *safetensors.File) (*StemConv2D, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron ASR checkpoint")
	}
	weight, wshape, err := file.GetFloat32("encoder.subsampling.conv_in.weight")
	if err != nil {
		return nil, fmt.Errorf("Nemotron ASR stem weight: %w", err)
	}
	bias, bshape, err := file.GetFloat32("encoder.subsampling.conv_in.bias")
	if err != nil {
		return nil, fmt.Errorf("Nemotron ASR stem bias: %w", err)
	}
	if len(wshape) != 4 || wshape[0] != stemChannels || wshape[1] != 1 || wshape[2] != 3 || wshape[3] != 3 || len(weight) != stemChannels*9 || len(bshape) != 1 || bshape[0] != stemChannels || len(bias) != stemChannels {
		return nil, fmt.Errorf("invalid Nemotron ASR stem checkpoint shape")
	}
	for _, v := range weight {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR stem weight")
		}
	}
	for _, v := range bias {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR stem bias")
		}
	}
	return &StemConv2D{Weight: weight, Bias: bias}, nil
}

// StemScratch holds per-call SIMD temporaries; it must not be shared by
// concurrent calls. Returned outputs remain owned, never views into scratch.
type StemScratch struct {
	patches []float32
}

// ForwardOffline consumes frame-major [frames,128] features and returns owned
// channel-major [256, floor(frames/2)+1, 65] pre-activation values. Reference
// padding is time (2,1), frequency (2,1), kernel (3,3), stride (2,2).
func (s *StemConv2D) ForwardOffline(features []float32, frames int) ([]float32, error) {
	return s.forward(features, frames, true, nil)
}

// ForwardOfflineScratch preserves output ownership while reusing per-session
// temporaries. A nil scratch has the same behaviour as ForwardOffline.
func (s *StemConv2D) ForwardOfflineScratch(features []float32, frames int, scratch *StemScratch) ([]float32, error) {
	return s.forward(features, frames, true, scratch)
}
func (s *StemConv2D) ForwardOfflineScalar(features []float32, frames int) ([]float32, error) {
	return s.forward(features, frames, false, nil)
}
func (s *StemConv2D) forward(features []float32, frames int, vector bool, scratch *StemScratch) ([]float32, error) {
	if s == nil || len(s.Weight) != stemChannels*9 || len(s.Bias) != stemChannels {
		return nil, fmt.Errorf("invalid Nemotron ASR stem weights")
	}
	if frames < 1 || frames > 3001 || len(features) != frames*128 {
		return nil, fmt.Errorf("invalid Nemotron ASR stem feature shape")
	}
	for _, v := range features {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite Nemotron ASR stem feature")
		}
	}
	rows := frames/2 + 1
	const freq = 65
	out := make([]float32, stemChannels*rows*freq)
	// Implicit zero padding avoids materializing a padded audio tensor.
	if !vector || !simd.HasSgemmAsm {
		for ch := 0; ch < stemChannels; ch++ {
			for row := 0; row < rows; row++ {
				for col := 0; col < freq; col++ {
					value := s.Bias[ch]
					for kt := 0; kt < 3; kt++ {
						sourceRow := row*2 + kt - 2
						if sourceRow < 0 || sourceRow >= frames {
							continue
						}
						for kf := 0; kf < 3; kf++ {
							sourceCol := col*2 + kf - 2
							if sourceCol < 0 || sourceCol >= 128 {
								continue
							}
							value += features[sourceRow*128+sourceCol] * s.Weight[ch*9+kt*3+kf]
						}
					}
					out[ch*rows*freq+row*freq+col] = value
				}
			}
		}
		return out, nil
	}
	// Patch-major [rows*65,9] and weight [256,9] allow a checked SIMD
	// projection. Transpose once to the PyTorch channel-major output contract.
	patchCount := rows * freq * 9
	var patches []float32
	if scratch == nil {
		patches = make([]float32, patchCount)
	} else {
		if cap(scratch.patches) < patchCount {
			scratch.patches = make([]float32, patchCount)
		}
		patches = scratch.patches[:patchCount]
		clear(patches) // Boundary patches must not retain a previous call's values.
	}
	for row := 0; row < rows; row++ {
		for col := 0; col < freq; col++ {
			p := patches[(row*freq+col)*9:]
			for kt := 0; kt < 3; kt++ {
				r := row*2 + kt - 2
				if r < 0 || r >= frames {
					continue
				}
				for kf := 0; kf < 3; kf++ {
					f := col*2 + kf - 2
					if f >= 0 && f < 128 {
						p[kt*3+kf] = features[r*128+f]
					}
				}
			}
		}
	}
	// Project weight [256,9] against patch rows [positions,9]. Keeping
	// channels as the GEMM rows writes directly to owned channel-major output,
	// avoiding a second full output tensor and a transpose pass.
	positions := rows * freq
	if !simd.DenseNTTo(out, s.Weight, patches, stemChannels, positions, 9, 1, 9, 9, positions) {
		return nil, fmt.Errorf("Nemotron ASR stem checked projection rejected shape")
	}
	for ch := 0; ch < stemChannels; ch++ {
		for i := range out[ch*positions : (ch+1)*positions] {
			out[ch*positions+i] += s.Bias[ch]
		}
	}
	return out, nil
}
