// Package fft provides FFT primitives for audio processing.
// This file contains the scalar fallback implementation.
package fft

import "math"

// ForwardReal computes the real-input FFT of n samples, returning n/2+1 complex bins
// as interleaved [real0, imag0, real1, imag1, ...].
// Input length must be a power of 2.
func ForwardReal(input []float32) []float32 {
	n := len(input)
	if n == 0 || n&(n-1) != 0 {
		return nil
	}

	// Convert to complex for in-place FFT
	x := make([]float64, 2*n)
	for i, v := range input {
		x[2*i] = float64(v)
		x[2*i+1] = 0
	}

	// In-place radix-2 FFT
	fftComplex(x, n)

	// Return first n/2+1 bins as float32 interleaved
	bins := n/2 + 1
	out := make([]float32, bins*2)
	for i := 0; i < bins; i++ {
		out[2*i] = float32(x[2*i])
		out[2*i+1] = float32(x[2*i+1])
	}
	return out
}

// PowerSpectrum computes |FFT(input)|² for each bin, returning n/2+1 power values.
func PowerSpectrum(input []float32) []float32 {
	bins := ForwardReal(input)
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

// fftComplex performs in-place radix-2 Cooley-Tukey FFT on interleaved complex data.
// x contains 2*n float64 values: [re0, im0, re1, im1, ...]
func fftComplex(x []float64, n int) {
	// Bit-reversal permutation
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j ^= bit
		if i < j {
			x[2*i], x[2*j] = x[2*j], x[2*i]
			x[2*i+1], x[2*j+1] = x[2*j+1], x[2*i+1]
		}
	}

	// Butterfly stages
	for size := 2; size <= n; size <<= 1 {
		half := size / 2
		angle := -2 * math.Pi / float64(size)
		for start := 0; start < n; start += size {
			for k := 0; k < half; k++ {
				theta := angle * float64(k)
				wr := math.Cos(theta)
				wi := math.Sin(theta)

				aIdx := 2 * (start + k)
				bIdx := 2 * (start + k + half)

				tr := wr*x[bIdx] - wi*x[bIdx+1]
				ti := wr*x[bIdx+1] + wi*x[bIdx]

				x[bIdx] = x[aIdx] - tr
				x[bIdx+1] = x[aIdx+1] - ti
				x[aIdx] += tr
				x[aIdx+1] += ti
			}
		}
	}
}
