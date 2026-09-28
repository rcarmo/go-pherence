package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const subsamplingChannels = 256

type subsamplingStage struct{ depthWeight, depthBias, pointWeight, pointBias []float32 }

// Subsampling owns released offline Conv2D and projection weights. Streaming
// caches and the 24-layer ASR encoder are separate, unimplemented contracts.
type Subsampling struct {
	Stem                     *StemConv2D
	layers                   [2]subsamplingStage
	linearWeight, linearBias []float32
}

func LoadSubsampling(file *safetensors.File) (*Subsampling, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron ASR checkpoint")
	}
	stem, err := LoadStemConv2D(file)
	if err != nil {
		return nil, err
	}
	load := func(name string, shape ...int) ([]float32, error) {
		values, got, e := file.GetFloat32(name)
		if e != nil {
			return nil, fmt.Errorf("%s: %w", name, e)
		}
		if len(got) != len(shape) {
			return nil, fmt.Errorf("%s invalid rank", name)
		}
		size := 1
		for i, d := range shape {
			if got[i] != d {
				return nil, fmt.Errorf("%s invalid shape %v", name, got)
			}
			size *= d
		}
		if len(values) != size {
			return nil, fmt.Errorf("%s invalid elements", name)
		}
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("%s non-finite", name)
			}
		}
		return values, nil
	}
	m := &Subsampling{Stem: stem}
	for i := range m.layers {
		prefix := fmt.Sprintf("encoder.subsampling.layers.%d.", i)
		s := &m.layers[i]
		if s.depthWeight, err = load(prefix+"depthwise_conv.weight", 256, 1, 3, 3); err != nil {
			return nil, err
		}
		if s.depthBias, err = load(prefix+"depthwise_conv.bias", 256); err != nil {
			return nil, err
		}
		if s.pointWeight, err = load(prefix+"pointwise_conv.weight", 256, 256, 1, 1); err != nil {
			return nil, err
		}
		if s.pointBias, err = load(prefix+"pointwise_conv.bias", 256); err != nil {
			return nil, err
		}
	}
	if m.linearWeight, err = load("encoder.subsampling.linear.weight", 1024, 4352); err != nil {
		return nil, err
	}
	if m.linearBias, err = load("encoder.subsampling.linear.bias", 1024); err != nil {
		return nil, err
	}
	return m, nil
}

// ForwardOffline returns owned [rows,1024] embeddings, where rows follow
// three offline causal Conv2D strides: rows = rows/2+1 at each stage.
// The caller supplies the processor's valid frame count (<=frames), including
// its masked final STFT frame. Only 1..128 frames are admitted for this
// bounded reference slice; the 1101-frame full recording is not qualified.
func (m *Subsampling) ForwardOffline(features []float32, frames, valid int) ([]float32, error) {
	if m == nil || m.Stem == nil || len(m.linearWeight) != 1024*4352 || len(m.linearBias) != 1024 {
		return nil, fmt.Errorf("invalid Nemotron ASR subsampling model")
	}
	if frames < 1 || frames > 128 || valid < 0 || valid > frames || len(features) != frames*128 {
		return nil, fmt.Errorf("invalid Nemotron ASR subsampling input")
	}
	hidden, err := m.Stem.ForwardOffline(features, frames)
	if err != nil {
		return nil, err
	}
	rows, width := frames/2+1, 65
	valid = valid/2 + 1 // Offline causal Conv2D adds one output even for an even input length.
	maskActivate(hidden, rows, width, valid)
	for i := range m.layers {
		hidden, rows, width, valid, err = m.layers[i].forward(hidden, rows, width, valid)
		if err != nil {
			return nil, err
		}
	}
	if width*subsamplingChannels != 4352 {
		return nil, fmt.Errorf("unexpected Nemotron ASR flattened width")
	}
	flattened := make([]float32, rows*4352)
	for row := 0; row < rows; row++ {
		for ch := 0; ch < subsamplingChannels; ch++ {
			copy(flattened[row*4352+ch*width:row*4352+(ch+1)*width], hidden[ch*rows*width+row*width:ch*rows*width+(row+1)*width])
		}
	}
	out := make([]float32, rows*1024)
	if !simd.DenseNTTo(out, flattened, m.linearWeight, rows, 1024, 4352, 1, 4352, 4352, 1024) {
		return nil, fmt.Errorf("Nemotron ASR final projection rejected shape")
	}
	for row := 0; row < rows; row++ {
		for c := 0; c < 1024; c++ {
			out[row*1024+c] += m.linearBias[c]
		}
	}
	return out, nil
}

func maskActivate(values []float32, rows, width, valid int) {
	for ch := 0; ch < subsamplingChannels; ch++ {
		for row := 0; row < rows; row++ {
			for f := 0; f < width; f++ {
				i := ch*rows*width + row*width + f
				if row >= valid || values[i] < 0 {
					values[i] = 0
				}
			}
		}
	}
}

func (s *subsamplingStage) forward(input []float32, rows, width, valid int) ([]float32, int, int, int, error) {
	if len(s.depthWeight) != subsamplingChannels*9 || len(s.depthBias) != subsamplingChannels || len(s.pointWeight) != subsamplingChannels*subsamplingChannels || len(s.pointBias) != subsamplingChannels {
		return nil, 0, 0, 0, fmt.Errorf("invalid Nemotron ASR subsampling stage")
	}
	outRows, outWidth := rows/2+1, width/2+1
	positions := outRows * outWidth
	// Pointwise GEMM consumes position-major input. Write depthwise results
	// directly into that layout instead of allocating a channel-major copy.
	patch := make([]float32, positions*subsamplingChannels)
	for ch := 0; ch < subsamplingChannels; ch++ {
		for row := 0; row < outRows; row++ {
			for col := 0; col < outWidth; col++ {
				v := s.depthBias[ch]
				for kt := 0; kt < 3; kt++ {
					r := row*2 + kt - 2
					if r < 0 || r >= rows {
						continue
					}
					for kf := 0; kf < 3; kf++ {
						f := col*2 + kf - 2
						if f >= 0 && f < width {
							v += input[ch*rows*width+r*width+f] * s.depthWeight[ch*9+kt*3+kf]
						}
					}
				}
				patch[(row*outWidth+col)*subsamplingChannels+ch] = v
			}
		}
	}
	projected := make([]float32, positions*subsamplingChannels)
	if !simd.DenseNTTo(projected, patch, s.pointWeight, positions, subsamplingChannels, subsamplingChannels, 1, subsamplingChannels, subsamplingChannels, subsamplingChannels) {
		return nil, 0, 0, 0, fmt.Errorf("Nemotron ASR pointwise projection rejected shape")
	}
	output := make([]float32, len(patch))
	nextValid := valid/2 + 1
	for ch := 0; ch < subsamplingChannels; ch++ {
		for row := 0; row < outRows; row++ {
			for col := 0; col < outWidth; col++ {
				value := projected[(row*outWidth+col)*subsamplingChannels+ch] + s.pointBias[ch]
				if row >= nextValid || value < 0 {
					value = 0
				}
				output[ch*positions+row*outWidth+col] = value
			}
		}
	}
	return output, outRows, outWidth, nextValid, nil
}
