//go:build amd64

package fft

import "math"

// MelSpectrogramFused computes the mel spectrogram using fused window+FFT+mel+log per frame.
// This avoids intermediate allocations by reusing buffers across frames.
// samples: input audio (16kHz)
// window: precomputed Hann window [fftSize]
// filters: mel filterbank [numMels * numBins] (dense)
// output: [numMels * numFrames] (written in mel-major order)
func MelSpectrogramFused(output, samples, window []float32, filters []float32, numMels, numBins, fftSize, fftPadded, hopLength, numFrames int) {
	if numFrames <= 0 || fftPadded <= 0 || numMels <= 0 || numBins <= 0 {
		return
	}
	if len(output) < numMels*numFrames || len(window) < fftSize {
		return
	}

	// Reusable buffers per frame
	re := make([]float64, fftPadded)
	im := make([]float64, fftPadded)
	power := make([]float32, numBins)

	for frame := 0; frame < numFrames; frame++ {
		offset := frame * hopLength

		// Step 1: Window + zero-pad into re/im
		for i := range re {
			re[i] = 0
			im[i] = 0
		}
		for i := 0; i < fftSize && offset+i < len(samples); i++ {
			re[i] = float64(samples[offset+i]) * float64(window[i])
		}

		// Step 2: In-place FFT
		bitReverse(re, im, fftPadded)
		// Butterfly with incremental twiddle (same as forwardRealOpt)
		for size := 2; size <= fftPadded; size <<= 1 {
			half := size / 2
			angle := -2 * math.Pi / float64(size)
			wRe := 1.0
			wIm := 0.0
			wnRe := math.Cos(angle)
			wnIm := math.Sin(angle)

			for k := 0; k < half; k++ {
				for start := 0; start < fftPadded; start += size {
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

		// Step 3: Power spectrum
		for k := 0; k < numBins; k++ {
			power[k] = float32(re[k]*re[k] + im[k]*im[k])
		}

		// Step 4: Mel filterbank + log (fused)
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
			output[m*numFrames+frame] = float32(math.Log10(energy))
		}
	}
}

// PrecomputeHannWindow returns a Hann window of length n.
func PrecomputeHannWindow(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n))))
	}
	return w
}

// PrecomputeMelFilters returns a dense mel filterbank [numMels * numBins].
func PrecomputeMelFilters(numMels, numBins, sampleRate, nfft int) []float32 {
	hzToMel := func(hz float64) float64 { return 2595 * math.Log10(1+hz/700) }
	melToHz := func(m float64) float64 { return 700 * (math.Pow(10, m/2595) - 1) }

	maxHz := float64(sampleRate) / 2
	melMin := hzToMel(0)
	melMax := hzToMel(maxHz)

	melPoints := make([]float64, numMels+2)
	for i := range melPoints {
		melPoints[i] = melMin + float64(i)*(melMax-melMin)/float64(numMels+1)
	}

	binPoints := make([]int, numMels+2)
	for i, m := range melPoints {
		hz := melToHz(m)
		binPoints[i] = int(math.Floor(hz * float64(nfft) / float64(sampleRate)))
	}

	filters := make([]float32, numMels*numBins)
	for m := 0; m < numMels; m++ {
		left := binPoints[m]
		center := binPoints[m+1]
		right := binPoints[m+2]
		for k := left; k < center && k < numBins; k++ {
			if center > left {
				filters[m*numBins+k] = float32(k-left) / float32(center-left)
			}
		}
		for k := center; k <= right && k < numBins; k++ {
			if right > center {
				filters[m*numBins+k] = float32(right-k) / float32(right-center)
			}
		}
	}
	return filters
}
