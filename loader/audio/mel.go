package audio

import (
	"math"

	"github.com/rcarmo/go-pherence/backends/simd/fft"
)

// MelConfig holds mel spectrogram parameters matching Whisper defaults.
type MelConfig struct {
	SampleRate int // Expected 16000
	FFTSize    int // 400 (25ms at 16kHz), zero-padded to NFFTPadded
	HopLength  int // 160 (10ms at 16kHz)
	NumMels    int // 80
	NFFTPadded int // Next power of 2 >= FFTSize (512)
}

// DefaultMelConfig returns Whisper's standard mel configuration.
func DefaultMelConfig() MelConfig {
	return MelConfig{
		SampleRate: 16000,
		FFTSize:    400,
		HopLength:  160,
		NumMels:    80,
		NFFTPadded: 512,
	}
}

// MelSpectrogram computes a log-mel spectrogram from audio samples.
// Returns mel[numMels][numFrames].
func MelSpectrogram(samples []float32, cfg MelConfig) [][]float32 {
	if len(samples) == 0 {
		return nil
	}

	numFrames := (len(samples) - cfg.FFTSize) / cfg.HopLength
	if numFrames <= 0 {
		numFrames = 1
	}

	// Pre-compute Hann window
	window := hannWindow(cfg.FFTSize)

	// Pre-compute mel filterbank
	numBins := cfg.NFFTPadded/2 + 1
	filters := melFilterbank(cfg.NumMels, numBins, cfg.SampleRate, cfg.NFFTPadded)

	// Allocate output
	mel := make([][]float32, cfg.NumMels)
	for i := range mel {
		mel[i] = make([]float32, numFrames)
	}

	// Process each frame
	frameBuf := make([]float32, cfg.NFFTPadded)
	for frame := 0; frame < numFrames; frame++ {
		offset := frame * cfg.HopLength

		// Apply window and zero-pad
		for i := range frameBuf {
			frameBuf[i] = 0
		}
		for i := 0; i < cfg.FFTSize && offset+i < len(samples); i++ {
			frameBuf[i] = samples[offset+i] * window[i]
		}

		// Power spectrum via optimized FFT
		power := fft.PowerSpectrumSIMD(frameBuf)

		// Apply mel filterbank
		for m := 0; m < cfg.NumMels; m++ {
			var energy float64
			for k := 0; k < numBins && k < len(power); k++ {
				if filters[m][k] == 0 {
					continue
				}
				energy += float64(filters[m][k]) * float64(power[k])
			}
			if energy < 1e-10 {
				energy = 1e-10
			}
			mel[m][frame] = float32(math.Log10(energy))
		}
	}

	// Whisper normalization: log10 mel, clamp to max-8dB, then (x+4)/4.
	// This matches OpenAI Whisper's log_mel_spectrogram normalization.
	if mel != nil {
		maxVal := mel[0][0]
		for _, row := range mel {
			for _, v := range row {
				if v > maxVal {
					maxVal = v
				}
			}
		}
		floor := maxVal - 8.0
		for m := range mel {
			for t := range mel[m] {
				v := mel[m][t]
				if v < floor {
					v = floor
				}
				mel[m][t] = (v + 4.0) / 4.0
			}
		}
	}

	return mel
}

// hannWindow computes a periodic Hann window of length n.
func hannWindow(n int) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n))))
	}
	return w
}

// melFilterbank computes triangular mel filterbank weights.
// Returns filters[numMels][numBins].
func melFilterbank(numMels, numBins, sampleRate, nfft int) [][]float32 {
	// Hz to mel and back
	hzToMel := func(hz float64) float64 { return 2595 * math.Log10(1+hz/700) }
	melToHz := func(m float64) float64 { return 700 * (math.Pow(10, m/2595) - 1) }

	maxHz := float64(sampleRate) / 2
	melMin := hzToMel(0)
	melMax := hzToMel(maxHz)

	// Equally spaced mel points
	melPoints := make([]float64, numMels+2)
	for i := range melPoints {
		melPoints[i] = melMin + float64(i)*(melMax-melMin)/float64(numMels+1)
	}

	// Convert to FFT bin indices
	binPoints := make([]int, numMels+2)
	for i, m := range melPoints {
		hz := melToHz(m)
		binPoints[i] = int(math.Floor(hz * float64(nfft) / float64(sampleRate)))
	}

	filters := make([][]float32, numMels)
	for m := 0; m < numMels; m++ {
		filters[m] = make([]float32, numBins)
		left := binPoints[m]
		center := binPoints[m+1]
		right := binPoints[m+2]

		for k := left; k < center && k < numBins; k++ {
			if center > left {
				filters[m][k] = float32(k-left) / float32(center-left)
			}
		}
		for k := center; k <= right && k < numBins; k++ {
			if right > center {
				filters[m][k] = float32(right-k) / float32(right-center)
			}
		}
	}
	return filters
}
