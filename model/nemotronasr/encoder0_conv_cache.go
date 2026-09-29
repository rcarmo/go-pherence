package nemotronasr

import (
	"fmt"
	"math"
)

// Encoder0ConvCache owns eight prior GLU frames per channel for the first ASR
// encoder convolution. It is single-stream mutable state, not safe for
// concurrent use. This does not include the attention key/value cache.
type Encoder0ConvCache struct {
	state []float32 // [1024,8] channel-major, oldest to newest
}

// Update accepts channel-major [1024,frames] GLU input, returns an owned
// [1024,8+frames] padded input and [1024,frames] depthwise result. On short
// chunks it retains the needed tail of the previous cache before appending
// new frames. Failed validation leaves the cache untouched.
func (c *Encoder0ConvCache) Update(glu []float32, frames int, depthWeight []float32) (padded, depth []float32, err error) {
	if c == nil || frames < 1 || frames > 5 || len(glu) != encoderWidth*frames || len(depthWeight) != encoderWidth*encoderConvKernel || (c.state != nil && len(c.state) != encoderWidth*(encoderConvKernel-1)) {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR encoder-0 convolution cache input")
	}
	for _, values := range [][]float32{glu, depthWeight} {
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, nil, fmt.Errorf("non-finite Nemotron ASR encoder-0 convolution cache input")
			}
		}
	}
	const left = encoderConvKernel - 1
	padded = make([]float32, encoderWidth*(left+frames))
	depth = make([]float32, encoderWidth*frames)
	next := make([]float32, encoderWidth*left)
	for ch := 0; ch < encoderWidth; ch++ {
		row := padded[ch*(left+frames) : (ch+1)*(left+frames)]
		if c.state != nil {
			copy(row[:left], c.state[ch*left:(ch+1)*left])
		}
		copy(row[left:], glu[ch*frames:(ch+1)*frames])
		copy(next[ch*left:(ch+1)*left], row[frames:frames+left])
		for t := 0; t < frames; t++ {
			var sum float32
			for tap := 0; tap < encoderConvKernel; tap++ {
				sum += row[t+tap] * depthWeight[ch*encoderConvKernel+tap]
			}
			depth[ch*frames+t] = sum
		}
	}
	c.state = next
	return padded, depth, nil
}

// Snapshot returns owned channel-major [1024,8] state for diagnostics or
// explicit stream handoff. Before the first update it returns zero state.
func (c *Encoder0ConvCache) Snapshot() []float32 {
	if c == nil {
		return nil
	}
	out := make([]float32, encoderWidth*(encoderConvKernel-1))
	copy(out, c.state)
	return out
}
