package speaker

import "math"

// SpeechBrainFbank computes the normalized 80-bin Fbank features expected by
// speechbrain/spkrec-ecapa-voxceleb. It mirrors the checkpoint hyperparams:
// 16 kHz audio, 25 ms window, 10 ms hop, n_fft=400, centered constant padding,
// power spectrogram, triangular HTK-mel filters, 10*log10 with top_db=80, and
// sentence mean normalization with std_norm=false. It returns [80][frames].
func SpeechBrainFbank(samples []float32, sampleRate int) [][]float32 {
	if len(samples) == 0 || sampleRate <= 0 {
		return nil
	}
	if sampleRate != 16000 {
		// Keep this function deterministic and dependency-free. Callers that load
		// non-16k audio should resample before invoking it.
		return nil
	}
	const (
		nMels  = 80
		nFFT   = 400
		winLen = 400
		hop    = 160
	)
	pad := nFFT / 2
	padded := make([]float32, len(samples)+2*pad)
	copy(padded[pad:], samples)
	frames := 1 + (len(padded)-nFFT)/hop
	if frames <= 0 {
		frames = 1
	}
	window := hammingWindow(winLen)
	filters := speechBrainMelFilters(nMels, nFFT/2+1, 16000, nFFT)
	features := make([][]float32, nMels)
	for m := range features {
		features[m] = make([]float32, frames)
	}
	frameBuf := make([]float32, nFFT)
	for t := 0; t < frames; t++ {
		off := t * hop
		for i := 0; i < nFFT; i++ {
			frameBuf[i] = 0
			if off+i < len(padded) && i < winLen {
				frameBuf[i] = padded[off+i] * window[i]
			}
		}
		power := dftPowerSpectrum(frameBuf)
		for m := 0; m < nMels; m++ {
			var e float64
			for k := 0; k < len(power); k++ {
				e += float64(power[k]) * float64(filters[m][k])
			}
			if e < 1e-10 {
				e = 1e-10
			}
			features[m][t] = float32(10 * math.Log10(e))
		}
	}
	// SpeechBrain top_db clamp over full utterance.
	maxVal := features[0][0]
	for _, row := range features {
		for _, v := range row {
			if v > maxVal {
				maxVal = v
			}
		}
	}
	floor := maxVal - 80
	for m := range features {
		var mean float32
		for t := range features[m] {
			if features[m][t] < floor {
				features[m][t] = floor
			}
			mean += features[m][t]
		}
		mean /= float32(frames)
		for t := range features[m] {
			features[m][t] -= mean
		}
	}
	return features
}

func hammingWindow(n int) []float32 {
	w := make([]float32, n)
	if n <= 1 {
		return w
	}
	for i := range w {
		w[i] = float32(0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n-1)))
	}
	return w
}

func dftPowerSpectrum(input []float32) []float32 {
	n := len(input)
	bins := n/2 + 1
	out := make([]float32, bins)
	for k := 0; k < bins; k++ {
		var re, im float64
		for t, v := range input {
			angle := -2 * math.Pi * float64(k*t) / float64(n)
			re += float64(v) * math.Cos(angle)
			im += float64(v) * math.Sin(angle)
		}
		out[k] = float32(re*re + im*im)
	}
	return out
}

func speechBrainMelFilters(numMels, numBins, sampleRate, nFFT int) [][]float32 {
	hzToMel := func(hz float64) float64 { return 2595 * math.Log10(1+hz/700) }
	melToHz := func(mel float64) float64 { return 700 * (math.Pow(10, mel/2595) - 1) }
	melMin := hzToMel(0)
	melMax := hzToMel(float64(sampleRate) / 2)
	hz := make([]float64, numMels+2)
	for i := range hz {
		mel := melMin + float64(i)*(melMax-melMin)/float64(numMels+1)
		hz[i] = melToHz(mel)
	}
	band := make([]float64, numMels)
	center := make([]float64, numMels)
	for i := 0; i < numMels; i++ {
		band[i] = hz[i+1] - hz[i]
		center[i] = hz[i+1]
	}
	filters := make([][]float32, numMels)
	for m := 0; m < numMels; m++ {
		filters[m] = make([]float32, numBins)
		for k := 0; k < numBins; k++ {
			freq := float64(k) * float64(sampleRate) / float64(nFFT)
			slope := (freq - center[m]) / band[m]
			left := slope + 1
			right := -slope + 1
			v := math.Min(left, right)
			if v < 0 {
				v = 0
			}
			filters[m][k] = float32(v)
		}
	}
	return filters
}
