//go:build amd64

package fft

import (
	"math"
	"testing"
)

func TestMelSpectrogramFused(t *testing.T) {
	// 1 second of 1kHz tone at 16kHz
	sampleRate := 16000
	samples := make([]float32, sampleRate)
	for i := range samples {
		samples[i] = float32(math.Sin(2 * math.Pi * 1000 * float64(i) / float64(sampleRate)))
	}

	fftSize := 400
	fftPadded := 512
	hopLength := 160
	numMels := 80
	numBins := fftPadded/2 + 1
	numFrames := (len(samples) - fftSize) / hopLength

	window := PrecomputeHannWindow(fftSize)
	filters := PrecomputeMelFilters(numMels, numBins, sampleRate, fftPadded)
	output := make([]float32, numMels*numFrames)

	MelSpectrogramFused(output, samples, window, filters, numMels, numBins, fftSize, fftPadded, hopLength, numFrames)

	// Find peak mel bin in first frame
	maxBin := 0
	maxVal := output[0]
	for m := 1; m < numMels; m++ {
		if output[m*numFrames] > maxVal {
			maxVal = output[m*numFrames]
			maxBin = m
		}
	}
	// 1kHz should fall in mel bin ~15-30
	if maxBin < 10 || maxBin > 40 {
		t.Fatalf("1kHz peak at mel bin %d (expected 15-30)", maxBin)
	}
	t.Logf("1kHz peak at mel bin %d, value %.2f", maxBin, maxVal)
}

func BenchmarkMelSpectrogramFused_3s(b *testing.B) {
	sampleRate := 16000
	samples := make([]float32, 3*sampleRate) // 3 seconds
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
		MelSpectrogramFused(output, samples, window, filters, numMels, numBins, fftSize, fftPadded, hopLength, numFrames)
	}
}

func BenchmarkMelSpectrogramFused_30s(b *testing.B) {
	sampleRate := 16000
	samples := make([]float32, 30*sampleRate) // 30 seconds (full Whisper chunk)
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
		MelSpectrogramFused(output, samples, window, filters, numMels, numBins, fftSize, fftPadded, hopLength, numFrames)
	}
}
