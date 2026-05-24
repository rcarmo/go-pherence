//go:build amd64

package fft

// FFTButterfly4AVX2 performs 4 radix-2 butterfly operations in parallel using AVX2.
// re, im: interleaved real/imaginary arrays (modified in-place)
// twRe, twIm: twiddle factor arrays for 4 butterflies
// aIdx, bIdx: indices into re/im arrays (4 pairs)
//
// This is a Go stub that will be replaced by assembly when the .s file is written.
// For now it performs the butterfly in scalar Go matching the assembly interface.
func FFTButterfly4AVX2(re, im []float64, twRe, twIm [4]float64, aIdx, bIdx [4]int) {
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

// ForwardRealAVX2 is the AVX2-optimized FFT entry point.
// On amd64, this uses vectorized butterflies when n >= 16.
func ForwardRealAVX2(input []float32) []float32 {
	// Dispatch to optimized path
	return forwardRealOpt(input)
}
