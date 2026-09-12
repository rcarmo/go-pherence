package fft

import "math"

// PrecomputeHannWindow returns the periodic Hann table used by the legacy
// padded-FFT mel path. It is architecture-independent; SIMD kernels consume the
// same values on supported architectures.
func PrecomputeHannWindow(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n))))
	}
	return w
}

// PrecomputeMelFilters returns the legacy dense HTK mel filterbank
// [numMels*numBins]. Model-specific exact Whisper/WeSpeaker frontends use their
// own checked filter contracts.
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
