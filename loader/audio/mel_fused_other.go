//go:build !amd64

package audio

func melSpectrogramFused(samples []float32, cfg MelConfig, numFrames int, filters [][]float32) ([][]float32, bool) {
	return nil, false
}
