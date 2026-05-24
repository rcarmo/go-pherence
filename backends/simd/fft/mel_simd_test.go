//go:build amd64

package fft

import "testing"

func BenchmarkMelSpectrogramSIMD_30s(b *testing.B) {
	sampleRate := 16000
	samples := make([]float32, 30*sampleRate)
	fftSize := 400
	fftPadded := 512
	hopLength := 160
	numMels := 80
	numBins := fftPadded/2 + 1
	numFrames := (len(samples) - fftSize) / hopLength

	window := PrecomputeHannWindow(fftSize)
	filters := PrecomputeMelFilters(numMels, numBins, sampleRate, fftPadded)
	output := make([]float32, numMels*numFrames)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MelSpectrogramSIMD(output, samples, window, filters, numMels, numBins, fftSize, fftPadded, hopLength, numFrames)
	}
}
