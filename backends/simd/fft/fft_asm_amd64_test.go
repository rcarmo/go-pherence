//go:build amd64

package fft

import (
	"math"
	"testing"
)

func TestForwardRealASM(t *testing.T) {
	n := 512
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Cos(2 * math.Pi * 10 * float64(i) / float64(n)))
	}

	out := ForwardRealASM(input)
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

func TestForwardRealASMMatchesOpt(t *testing.T) {
	// Verify ASM path produces same results as optimized Go path
	n := 256
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(math.Sin(2*math.Pi*7*float64(i)/float64(n)) + 0.5*math.Cos(2*math.Pi*23*float64(i)/float64(n)))
	}

	opt := forwardRealOpt(input)
	asm := ForwardRealASM(input)

	if len(opt) != len(asm) {
		t.Fatalf("length mismatch: opt=%d asm=%d", len(opt), len(asm))
	}

	maxDiff := float32(0)
	for i := range opt {
		diff := opt[i] - asm[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	// Allow small numerical differences from sin/cos approximation
	if maxDiff > 0.01 {
		t.Fatalf("max difference=%f (want < 0.01)", maxDiff)
	}
	t.Logf("max diff between opt and asm: %e", maxDiff)
}

func BenchmarkFFT512ASM(b *testing.B) {
	input := make([]float32, 512)
	for i := range input {
		input[i] = float32(i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ForwardRealASM(input)
	}
}
