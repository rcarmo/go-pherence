//go:build arm64

package fft

// FFTButterfly4NEON performs 4 radix-2 butterfly operations using NEON.
// This is a Go stub matching the NEON assembly interface.
func FFTButterfly4NEON(re, im []float64, twRe, twIm [4]float64, aIdx, bIdx [4]int) {
	for i := 0; i < 4; i++ {
		a := aIdx[i]
		b := bIdx[i]
		if a < 0 || a >= len(re) || b < 0 || b >= len(re) {
			continue
		}

		tr := twRe[i]*re[b] - twIm[i]*im[b]
		ti := twRe[i]*im[b] + twIm[i]*re[b]

		re[b] = re[a] - tr
		im[b] = im[a] - ti
		re[a] += tr
		im[a] += ti
	}
}

// ForwardRealNEON is the NEON-optimized FFT entry point.
func ForwardRealNEON(input []float32) []float32 {
	return forwardRealOpt(input)
}
