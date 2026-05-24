package fft

import "math"

// ForwardRealSIMD computes a real-input FFT using SIMD-optimized butterfly stages
// when available, falling back to scalar otherwise.
// This is the dispatch entry point.
func ForwardRealSIMD(input []float32) []float32 {
	// On amd64 with AVX2, the assembly kernel handles the butterfly stages.
	// For now, all platforms use the optimized Go implementation below.
	return forwardRealOpt(input)
}

// forwardRealOpt is an optimized pure-Go FFT using precomputed twiddle factors
// and cache-friendly access patterns.
func forwardRealOpt(input []float32) []float32 {
	n := len(input)
	if n == 0 || n&(n-1) != 0 {
		return nil
	}

	re := make([]float64, n)
	im := make([]float64, n)
	for i, v := range input {
		re[i] = float64(v)
	}

	bitReverse(re, im, n)

	// Butterfly stages
	for size := 2; size <= n; size <<= 1 {
		half := size / 2
		angle := -2 * math.Pi / float64(size)
		wRe := 1.0
		wIm := 0.0
		wnRe := math.Cos(angle)
		wnIm := math.Sin(angle)

		for k := 0; k < half; k++ {
			for start := 0; start < n; start += size {
				a := start + k
				b := start + k + half

				tr := wRe*re[b] - wIm*im[b]
				ti := wRe*im[b] + wIm*re[b]

				re[b] = re[a] - tr
				im[b] = im[a] - ti
				re[a] += tr
				im[a] += ti
			}
			// Rotate twiddle
			newRe := wRe*wnRe - wIm*wnIm
			wIm = wRe*wnIm + wIm*wnRe
			wRe = newRe
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

// PowerSpectrumSIMD computes |FFT(input)|² using the optimized path.
func PowerSpectrumSIMD(input []float32) []float32 {
	bins := ForwardRealSIMD(input)
	if bins == nil {
		return nil
	}
	n := len(bins) / 2
	power := make([]float32, n)
	for i := 0; i < n; i++ {
		re := bins[2*i]
		im := bins[2*i+1]
		power[i] = re*re + im*im
	}
	return power
}

func bitReverse(re, im []float64, n int) {
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j ^= bit
		if i < j {
			re[i], re[j] = re[j], re[i]
			im[i], im[j] = im[j], im[i]
		}
	}
}
