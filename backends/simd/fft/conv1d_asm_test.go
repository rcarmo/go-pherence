//go:build amd64

package fft

import "testing"

func TestConv1dK3S1InnerASM(t *testing.T) {
	// Test the ASM inner loop: identity kernel [0,1,0]
	n := 16
	input := make([]float32, n)
	out := make([]float32, n)
	for i := range input {
		input[i] = float32(i + 1)
	}

	conv1dK3S1Inner(&out[0], &input[0], 0, 1, 0, n)

	// Middle elements should match input (identity kernel)
	for i := 1; i < n-1; i++ {
		if out[i] != input[i] {
			t.Fatalf("out[%d]=%f want %f", i, out[i], input[i])
		}
	}
	// Boundary: out[0] = 0*0 + 1*input[0] + 0*input[1] = input[0]
	if out[0] != input[0] {
		t.Fatalf("out[0]=%f want %f", out[0], input[0])
	}
}

func TestConv1dK3S1InnerSum(t *testing.T) {
	// Sum kernel [1,1,1]
	n := 8
	input := make([]float32, n)
	out := make([]float32, n)
	for i := range input {
		input[i] = 1
	}

	conv1dK3S1Inner(&out[0], &input[0], 1, 1, 1, n)

	// Middle: should be 3
	for i := 1; i < n-1; i++ {
		if out[i] != 3 {
			t.Fatalf("out[%d]=%f want 3", i, out[i])
		}
	}
	// Boundaries: 2 (missing one neighbor)
	if out[0] != 2 {
		t.Fatalf("out[0]=%f want 2", out[0])
	}
	if out[n-1] != 2 {
		t.Fatalf("out[%d]=%f want 2", n-1, out[n-1])
	}
}

func BenchmarkConv1dK3S1Inner_480(b *testing.B) {
	n := 480
	input := make([]float32, n)
	out := make([]float32, n)
	for i := range input {
		input[i] = float32(i) * 0.01
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		conv1dK3S1Inner(&out[0], &input[0], 0.5, 1.0, 0.5, n)
	}
}
