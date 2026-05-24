package fft

import (
	"math"
	"testing"
)

func TestForwardRealSIMDCosine(t *testing.T) {
	n := 512
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Cos(2 * math.Pi * 10 * float64(i) / float64(n)))
	}

	out := ForwardRealSIMD(input)
	if out == nil {
		t.Fatal("nil")
	}

	// Bin 10 should have peak
	maxBin := 0
	maxMag := float32(0)
	for i := 0; i < len(out)/2; i++ {
		mag := out[2*i]*out[2*i] + out[2*i+1]*out[2*i+1]
		if mag > maxMag {
			maxMag = mag
			maxBin = i
		}
	}
	if maxBin != 10 {
		t.Fatalf("peak at bin %d want 10", maxBin)
	}
}

func TestPowerSpectrumSIMD(t *testing.T) {
	n := 256
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Cos(2 * math.Pi * 7 * float64(i) / float64(n)))
	}

	power := PowerSpectrumSIMD(input)
	if len(power) != n/2+1 {
		t.Fatalf("power length=%d want %d", len(power), n/2+1)
	}

	maxBin := 0
	maxVal := power[0]
	for i, v := range power {
		if v > maxVal {
			maxVal = v
			maxBin = i
		}
	}
	if maxBin != 7 {
		t.Fatalf("max power at bin %d want 7", maxBin)
	}
}

func BenchmarkFFT512(b *testing.B) {
	input := make([]float32, 512)
	for i := range input {
		input[i] = float32(i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ForwardRealSIMD(input)
	}
}

func BenchmarkFFT512Scalar(b *testing.B) {
	input := make([]float32, 512)
	for i := range input {
		input[i] = float32(i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ForwardReal(input)
	}
}

func BenchmarkPowerSpectrum512(b *testing.B) {
	input := make([]float32, 512)
	for i := range input {
		input[i] = float32(i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		PowerSpectrumSIMD(input)
	}
}
