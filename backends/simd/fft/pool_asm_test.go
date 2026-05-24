//go:build amd64

package fft

import (
	"math"
	"testing"
)

func TestMeanStdReduce(t *testing.T) {
	input := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	var mean, std float32
	meanStdReduce(&mean, &std, &input[0], len(input))

	expectedMean := float32(4.5)
	if math.Abs(float64(mean-expectedMean)) > 0.01 {
		t.Fatalf("mean=%f want %f", mean, expectedMean)
	}

	// std = sqrt(mean(x^2) - mean^2) = sqrt(21.25 - 20.25) = sqrt(5.25) ≈ 2.29
	expectedStd := float32(math.Sqrt(5.25))
	if math.Abs(float64(std-expectedStd)) > 0.01 {
		t.Fatalf("std=%f want %f", std, expectedStd)
	}
}

func TestMeanStdReduceLarge(t *testing.T) {
	// Test with >8 elements to exercise AVX path
	n := 64
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(i)
	}
	var mean, std float32
	meanStdReduce(&mean, &std, &input[0], n)

	expectedMean := float32(31.5) // mean of 0..63
	if math.Abs(float64(mean-expectedMean)) > 0.1 {
		t.Fatalf("mean=%f want %f", mean, expectedMean)
	}
	if std <= 0 {
		t.Fatalf("std=%f should be positive", std)
	}
	t.Logf("n=%d mean=%f std=%f", n, mean, std)
}

func BenchmarkMeanStdReduce_1536(b *testing.B) {
	input := make([]float32, 1536)
	for i := range input {
		input[i] = float32(i) * 0.01
	}
	var mean, std float32
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		meanStdReduce(&mean, &std, &input[0], 1536)
	}
}
