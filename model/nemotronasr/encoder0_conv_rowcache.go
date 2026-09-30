package nemotronasr

import (
	"fmt"
	"math"
)

// updateRowMajor computes the same causal depthwise operation directly from
// row-major GLU output and the previous channel-major history. Unlike public
// Update, the private operator does not materialise a padded input or transpose
// its result. dst is per-attempt scratch. State still commits by replacement.
func (c *Encoder0ConvCache) updateRowMajor(glu []float32, frames int, weights, dst []float32) error {
	if c == nil || frames < 1 || frames > 5 || len(glu) != frames*encoderWidth || len(weights) != encoderWidth*encoderConvKernel || len(dst) != len(glu) || c.state != nil && len(c.state) != encoderWidth*(encoderConvKernel-1) {
		return fmt.Errorf("invalid Nemotron ASR row-major convolution input")
	}
	for _, input := range [][]float32{glu, weights} {
		for _, v := range input {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("non-finite Nemotron ASR row-major convolution input")
			}
		}
	}
	const history = encoderConvKernel - 1
	next := make([]float32, encoderWidth*history)
	for row := 0; row < frames; row++ {
		for ch := 0; ch < encoderWidth; ch++ {
			var sum float32
			for tap := 0; tap < encoderConvKernel; tap++ {
				source := row + tap - history
				var value float32
				if source < 0 {
					if c.state != nil {
						value = c.state[ch*history+source+history]
					}
				} else {
					value = glu[source*encoderWidth+ch]
				}
				sum += value * weights[ch*encoderConvKernel+tap]
			}
			dst[row*encoderWidth+ch] = sum
		}
	}
	for ch := 0; ch < encoderWidth; ch++ {
		for h := 0; h < history; h++ {
			source := frames + h - history
			if source < 0 {
				if c.state != nil {
					next[ch*history+h] = c.state[ch*history+source+history]
				}
			} else {
				next[ch*history+h] = glu[source*encoderWidth+ch]
			}
		}
	}
	c.state = next
	return nil
}
