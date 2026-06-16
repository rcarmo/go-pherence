package whisper

import (
	nv "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	simdfft "github.com/rcarmo/go-pherence/backends/simd/fft"
	"github.com/rcarmo/go-pherence/loader/audio"
)

// MelFlatFromSamples computes Whisper log-mel features and flattens them in
// channel-major [numMels,T] layout. CPU/SIMD is the default oracle; setting
// GO_PHERENCE_WHISPER_GPU_MEL=1 tries the correctness-first CUDA/PTX mel body
// and falls back to CPU/SIMD on any allocation/upload/launch/download issue.
func MelFlatFromSamples(samples []float32, cfg Config) ([]float32, int) {
	melCfg := audio.MelConfig{SampleRate: 16000, FFTSize: 400, HopLength: 160, NumMels: cfg.NumMelBins, NFFTPadded: 512}
	if flat, frames, ok := melFlatGPU(samples, melCfg); ok {
		return flat, frames
	}
	mel := audio.MelSpectrogram(samples, melCfg)
	if len(mel) == 0 || len(mel[0]) == 0 {
		return nil, 0
	}
	return flattenMel(mel, cfg.NumMelBins)
}

func flattenMel(mel [][]float32, numMels int) ([]float32, int) {
	if len(mel) == 0 || len(mel[0]) == 0 || numMels <= 0 {
		return nil, 0
	}
	T := len(mel[0])
	flat := make([]float32, numMels*T)
	for m := 0; m < numMels && m < len(mel); m++ {
		copy(flat[m*T:(m+1)*T], mel[m])
	}
	return flat, T
}

func melFlatGPU(samples []float32, cfg audio.MelConfig) ([]float32, int, bool) {
	if !whisperGPUFeatureEnabled("GO_PHERENCE_WHISPER_GPU_MEL") || len(samples) == 0 || cfg.SampleRate != 16000 || cfg.FFTSize <= 0 || cfg.HopLength <= 0 || cfg.NumMels <= 0 || cfg.NFFTPadded != 512 || !nv.SgemmReady() {
		return nil, 0, false
	}
	numFrames := (len(samples) - cfg.FFTSize) / cfg.HopLength
	if numFrames <= 0 {
		numFrames = 1
	}
	numBins := cfg.NFFTPadded/2 + 1
	window := simdfft.PrecomputeHannWindow(cfg.FFTSize)
	filters := simdfft.PrecomputeMelFilters(cfg.NumMels, numBins, cfg.SampleRate, cfg.NFFTPadded)
	outLen := cfg.NumMels * numFrames

	audioBuf, err := nv.Malloc(len(samples))
	if err != nil {
		return nil, 0, false
	}
	defer audioBuf.Free()
	windowBuf, err := nv.Malloc(len(window))
	if err != nil {
		return nil, 0, false
	}
	defer windowBuf.Free()
	filterBuf, err := nv.Malloc(len(filters))
	if err != nil {
		return nil, 0, false
	}
	defer filterBuf.Free()
	outBuf, err := nv.Malloc(outLen)
	if err != nil {
		return nil, 0, false
	}
	defer outBuf.Free()
	if err := audioBuf.Upload(samples); err != nil {
		return nil, 0, false
	}
	if err := windowBuf.Upload(window); err != nil {
		return nil, 0, false
	}
	if err := filterBuf.Upload(filters); err != nil {
		return nil, 0, false
	}
	if err := nv.WhisperMelSpectrogramBuffer(outBuf, audioBuf, windowBuf, filterBuf, numFrames, cfg.FFTSize, cfg.HopLength, cfg.NumMels, numBins); err != nil {
		return nil, 0, false
	}
	flat := make([]float32, outLen)
	if err := outBuf.Download(flat); err != nil {
		return nil, 0, false
	}
	normalizeWhisperMelFlat(flat, cfg.NumMels, numFrames)
	return flat, numFrames, true
}

func normalizeWhisperMelFlat(mel []float32, numMels, frames int) {
	if len(mel) == 0 || numMels <= 0 || frames <= 0 {
		return
	}
	maxVal := mel[0]
	for _, v := range mel {
		if v > maxVal {
			maxVal = v
		}
	}
	floor := maxVal - 8.0
	for i, v := range mel {
		if v < floor {
			v = floor
		}
		mel[i] = (v + 4.0) / 4.0
	}
}
