package fft

import (
	"math"
	"unsafe"
)

// ForwardRealSIMD computes a real-input FFT using the optimized pure-Go
// butterfly stages. It does not dispatch to assembly.
func ForwardRealSIMD(input []float32) []float32 {
	return forwardRealOpt(input)
}

// forwardRealOpt is an optimized pure-Go FFT using precomputed twiddle factors
// and cache-friendly access patterns.
func forwardRealOpt(input []float32) []float32 {
	n := len(input)
	if n == 0 || n&(n-1) != 0 {
		return nil
	}

	re, im := make([]float64, n), make([]float64, n)
	out := make([]float32, (n/2+1)*2)
	if !ForwardRealInto(out, input, re, im) {
		return nil
	}
	return out
}

// ForwardRealInto writes the real FFT to caller-owned buffers. It rejects
// overlapping buffers; the caller may reuse the buffers across rows.
func ForwardRealInto(out, input []float32, re, im []float64) bool {
	n := len(input)
	if n == 0 || n&(n-1) != 0 || len(re) < n || len(im) < n || len(out) < (n/2+1)*2 {
		return false
	}
	re, im = re[:n], im[:n]
	out = out[:(n/2+1)*2]
	inputStart := uintptr(unsafe.Pointer(unsafe.SliceData(input)))
	outStart := uintptr(unsafe.Pointer(unsafe.SliceData(out)))
	reStart := uintptr(unsafe.Pointer(unsafe.SliceData(re)))
	imStart := uintptr(unsafe.Pointer(unsafe.SliceData(im)))
	overlaps := func(a, sizeA, b, sizeB uintptr) bool { return a < b+sizeB && b < a+sizeA }
	if overlaps(inputStart, uintptr(n)*4, outStart, uintptr(len(out))*4) ||
		overlaps(inputStart, uintptr(n)*4, reStart, uintptr(n)*8) ||
		overlaps(inputStart, uintptr(n)*4, imStart, uintptr(n)*8) ||
		overlaps(outStart, uintptr(len(out))*4, reStart, uintptr(n)*8) ||
		overlaps(outStart, uintptr(len(out))*4, imStart, uintptr(n)*8) ||
		overlaps(reStart, uintptr(n)*8, imStart, uintptr(n)*8) {
		return false
	}
	for i, v := range input {
		re[i] = float64(v)
	}
	clear(im)
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
	for i := 0; i < bins; i++ {
		out[2*i] = float32(re[i])
		out[2*i+1] = float32(im[i])
	}
	return true
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
