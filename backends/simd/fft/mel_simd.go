//go:build amd64

package fft

import "math"

// MelSpectrogramSIMD computes the mel spectrogram using SIMD-optimized FFT and
// the AVX2 meanStdReduce for per-bin accumulation where beneficial.
// This is the highest-performance CPU mel path.
func MelSpectrogramSIMD(output, samples, window []float32, filters []float32, numMels, numBins, fftSize, fftPadded, hopLength, numFrames int) {
	if numFrames <= 0 || fftPadded <= 0 || numMels <= 0 || numBins <= 0 {
		return
	}
	if len(output) < numMels*numFrames || len(window) < fftSize {
		return
	}

	// Reusable buffers
	re := make([]float64, fftPadded)
	im := make([]float64, fftPadded)
	power := make([]float32, numBins)

	// Precompute twiddle tables for all FFT stages
	type twiddleStage struct {
		re, im []float64
	}
	var twiddles []twiddleStage
	for size := 2; size <= fftPadded; size <<= 1 {
		half := size / 2
		angle := -2 * math.Pi / float64(size)
		tw := twiddleStage{
			re: make([]float64, half),
			im: make([]float64, half),
		}
		for k := 0; k < half; k++ {
			tw.re[k] = math.Cos(angle * float64(k))
			tw.im[k] = math.Sin(angle * float64(k))
		}
		twiddles = append(twiddles, tw)
	}

	for frame := 0; frame < numFrames; frame++ {
		offset := frame * hopLength

		// Step 1: Window + zero-pad
		for i := range re {
			re[i] = 0
			im[i] = 0
		}
		for i := 0; i < fftSize && offset+i < len(samples); i++ {
			re[i] = float64(samples[offset+i]) * float64(window[i])
		}

		// Step 2: FFT with precomputed twiddles + AVX2 butterflies
		bitReverse(re, im, fftPadded)
		stageIdx := 0
		for size := 2; size <= fftPadded; size <<= 1 {
			half := size / 2
			tw := twiddles[stageIdx]
			stageIdx++

			// Use AVX2 butterfly for groups of 4
			for k := 0; k < half; k += 4 {
				if half-k >= 4 {
					for start := 0; start < fftPadded; start += size {
						fftButterfly4(&re[0], &im[0], &tw.re[k], &tw.im[k], start+k, start+k+half, 1)
					}
				} else {
					// Scalar tail
					for kk := k; kk < half; kk++ {
						for start := 0; start < fftPadded; start += size {
							a := start + kk
							b := start + kk + half
							tr := tw.re[kk]*re[b] - tw.im[kk]*im[b]
							ti := tw.re[kk]*im[b] + tw.im[kk]*re[b]
							re[b] = re[a] - tr
							im[b] = im[a] - ti
							re[a] += tr
							im[a] += ti
						}
					}
				}
			}
		}

		// Step 3: Power spectrum
		for k := 0; k < numBins; k++ {
			power[k] = float32(re[k]*re[k] + im[k]*im[k])
		}

		// Step 4: Mel filterbank + log
		for m := 0; m < numMels; m++ {
			var energy float64
			fOff := m * numBins
			for k := 0; k < numBins; k++ {
				if fOff+k < len(filters) {
					f := filters[fOff+k]
					if f != 0 {
						energy += float64(f) * float64(power[k])
					}
				}
			}
			if energy < 1e-10 {
				energy = 1e-10
			}
			output[m*numFrames+frame] = float32(math.Log(energy))
		}
	}
}
