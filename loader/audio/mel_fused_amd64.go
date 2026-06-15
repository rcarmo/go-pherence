//go:build amd64

package audio

import "github.com/rcarmo/go-pherence/backends/simd/fft"

func melSpectrogramFused(samples []float32, cfg MelConfig, numFrames int, filters [][]float32) ([][]float32, bool) {
	if cfg.FFTSize <= 0 || cfg.NFFTPadded <= 0 || cfg.HopLength <= 0 || cfg.NumMels <= 0 || numFrames <= 0 {
		return nil, false
	}
	numBins := cfg.NFFTPadded/2 + 1
	if len(filters) < cfg.NumMels {
		return nil, false
	}
	dense := make([]float32, cfg.NumMels*numBins)
	for m := 0; m < cfg.NumMels; m++ {
		if len(filters[m]) < numBins {
			return nil, false
		}
		copy(dense[m*numBins:(m+1)*numBins], filters[m][:numBins])
	}
	window := hannWindow(cfg.FFTSize)
	out := make([]float32, cfg.NumMels*numFrames)
	fft.MelSpectrogramSIMD(out, samples, window, dense, cfg.NumMels, numBins, cfg.FFTSize, cfg.NFFTPadded, cfg.HopLength, numFrames)
	mel := make([][]float32, cfg.NumMels)
	for m := range mel {
		mel[m] = out[m*numFrames : (m+1)*numFrames]
	}
	return mel, true
}
