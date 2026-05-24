//go:build arm64

package fft

import "math"

// conv1dK3S1InnerNEON accumulates one input channel's Conv1D contribution using NEON.
//
//go:noescape
func conv1dK3S1InnerNEON(out, input *float32, w0, w1, w2 float32, n int)

// fftButterfly2NEON performs 2 parallel radix-2 butterflies using NEON.
//
//go:noescape
func fftButterfly2NEON(re, im *float64, twRe, twIm *float64, aOff, bOff int)

// ForwardRealNEON is the NEON-optimized FFT entry point.
func ForwardRealNEON(input []float32) []float32 {
	n := len(input)
	if n < 8 || n&(n-1) != 0 {
		return forwardRealOpt(input)
	}

	re := make([]float64, n)
	im := make([]float64, n)
	for i, v := range input {
		re[i] = float64(v)
	}

	bitReverse(re, im, n)

	for size := 2; size <= n; size <<= 1 {
		half := size / 2
		angle := -2 * math.Pi / float64(size)

		// Precompute twiddles
		tw_re := make([]float64, half)
		tw_im := make([]float64, half)
		for k := 0; k < half; k++ {
			tw_re[k] = math.Cos(angle * float64(k))
			tw_im[k] = math.Sin(angle * float64(k))
		}

		for k := 0; k < half; k += 2 {
			if half-k < 2 {
				// scalar tail
				for start := 0; start < n; start += size {
					a := start + k
					b := start + k + half
					tr := tw_re[k]*re[b] - tw_im[k]*im[b]
					ti := tw_re[k]*im[b] + tw_im[k]*re[b]
					re[b] = re[a] - tr
					im[b] = im[a] - ti
					re[a] += tr
					im[a] += ti
				}
				break
			}
			for start := 0; start < n; start += size {
				fftButterfly2NEON(&re[0], &im[0], &tw_re[k], &tw_im[k], start+k, start+k+half)
			}
		}
	}

	bins := n/2 + 1
	out := make([]float32, bins*2)
	for i := 0; i < bins; i++ {
		out[2*i] = float32(re[i])
		out[2*i+1] = float32(im[i])
	}
	return out
}
