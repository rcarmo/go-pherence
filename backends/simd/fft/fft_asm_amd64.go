//go:build amd64

package fft

import "math"

// meanStdReduce computes mean and std of a float32 slice using AVX2.
//
//go:noescape
func meanStdReduce(out_mean, out_std, input *float32, length int)

// fftButterfly4 performs 4 parallel radix-2 butterflies using AVX2 FMA.
// Go declaration for the assembly implementation.
//
//go:noescape
func fftButterfly4(re, im *float64, twRe, twIm *float64, aOff, bOff int, stride int)

// ForwardRealASM computes a real-input FFT using AVX2-accelerated butterfly stages.
// Falls back to forwardRealOpt if n < 16 or not aligned.
func ForwardRealASM(input []float32) []float32 {
	n := len(input)
	if n < 16 || n&(n-1) != 0 {
		return forwardRealOpt(input)
	}

	re := make([]float64, n)
	im := make([]float64, n)
	for i, v := range input {
		re[i] = float64(v)
	}

	bitReverse(re, im, n)

	// Use assembly butterfly for stages where half >= 4
	for size := 2; size <= n; size <<= 1 {
		half := size / 2
		if half >= 4 && len(re) >= n {
			// Process 4 butterflies at a time using assembly
			fftButterflyStageASM(re, im, n, size)
		} else {
			// Scalar for small stages
			fftButterflyStageScalar(re, im, n, size)
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

func fftButterflyStageASM(re, im []float64, n, size int) {
	half := size / 2
	angle := -2 * math.Pi / float64(size)

	// Precompute ALL twiddle factors for this stage
	allTwRe := make([]float64, half)
	allTwIm := make([]float64, half)
	for k := 0; k < half; k++ {
		allTwRe[k] = math.Cos(angle * float64(k))
		allTwIm[k] = math.Sin(angle * float64(k))
	}

	for k := 0; k < half; k += 4 {
		rem := half - k
		if rem < 4 {
			// Scalar tail
			for kk := k; kk < half; kk++ {
				for start := 0; start < n; start += size {
					a := start + kk
					b := start + kk + half
					tr := allTwRe[kk]*re[b] - allTwIm[kk]*im[b]
					ti := allTwRe[kk]*im[b] + allTwIm[kk]*re[b]
					re[b] = re[a] - tr
					im[b] = im[a] - ti
					re[a] += tr
					im[a] += ti
				}
			}
			break
		}

		// Apply 4 butterflies per group using AVX2 assembly
		for start := 0; start < n; start += size {
			fftButterfly4(&re[0], &im[0], &allTwRe[k], &allTwIm[k], start+k, start+k+half, 1)
		}
	}
}

func fftButterflyTailScalar(re, im []float64, n, size, kStart, half int, angle float64) {
	for k := kStart; k < half; k++ {
		theta := angle * float64(k)
		wr := cosApprox(theta)
		wi := sinApprox(theta)
		for start := 0; start < n; start += size {
			a := start + k
			b := start + k + half
			tr := wr*re[b] - wi*im[b]
			ti := wr*im[b] + wi*re[b]
			re[b] = re[a] - tr
			im[b] = im[a] - ti
			re[a] += tr
			im[a] += ti
		}
	}
}

func fftButterflyStageScalar(re, im []float64, n, size int) {
	half := size / 2
	angle := -2 * 3.141592653589793 / float64(size)
	wRe := 1.0
	wIm := 0.0
	wnRe := cosApprox(angle)
	wnIm := sinApprox(angle)

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
		newRe := wRe*wnRe - wIm*wnIm
		wIm = wRe*wnIm + wIm*wnRe
		wRe = newRe
	}
}

// Fast cos/sin — use standard library (well-optimized on amd64)
func cosApprox(x float64) float64 { return math.Cos(x) }
func sinApprox(x float64) float64 { return math.Sin(x) }
