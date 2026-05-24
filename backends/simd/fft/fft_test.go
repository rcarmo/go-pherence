package fft

import (
	"math"
	"testing"
)

func TestForwardRealDC(t *testing.T) {
	// All ones → DC bin has magnitude n, all others ~0
	n := 16
	input := make([]float32, n)
	for i := range input {
		input[i] = 1
	}
	out := ForwardReal(input)
	if out == nil {
		t.Fatal("ForwardReal returned nil")
	}
	// DC bin (index 0): real should be n, imag should be 0
	if math.Abs(float64(out[0])-float64(n)) > 0.01 {
		t.Fatalf("DC real=%f want %d", out[0], n)
	}
	if math.Abs(float64(out[1])) > 0.01 {
		t.Fatalf("DC imag=%f want 0", out[1])
	}
}

func TestForwardRealCosine(t *testing.T) {
	// Cosine at bin 3 of n=32
	n := 32
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Cos(2 * math.Pi * 3 * float64(i) / float64(n)))
	}
	out := ForwardReal(input)
	if out == nil {
		t.Fatal("nil")
	}
	// Bin 3 should have magnitude n/2 = 16
	re := out[2*3]
	im := out[2*3+1]
	mag := float32(math.Sqrt(float64(re*re + im*im)))
	if mag < 15.5 || mag > 16.5 {
		t.Fatalf("bin 3 magnitude=%f want ~16", mag)
	}
	// DC should be ~0
	if math.Abs(float64(out[0])) > 0.1 {
		t.Fatalf("DC=%f want ~0", out[0])
	}
}

func TestPowerSpectrum(t *testing.T) {
	n := 16
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Cos(2 * math.Pi * 2 * float64(i) / float64(n)))
	}
	power := PowerSpectrum(input)
	if len(power) != n/2+1 {
		t.Fatalf("power length=%d want %d", len(power), n/2+1)
	}
	// Bin 2 should have most energy
	maxBin := 0
	maxVal := power[0]
	for i, v := range power {
		if v > maxVal {
			maxVal = v
			maxBin = i
		}
	}
	if maxBin != 2 {
		t.Fatalf("max power at bin %d want 2", maxBin)
	}
}

func TestForwardRealNonPow2(t *testing.T) {
	// Non-power-of-2 should return nil
	input := make([]float32, 7)
	if out := ForwardReal(input); out != nil {
		t.Fatalf("expected nil for non-pow2, got %d elements", len(out))
	}
}
