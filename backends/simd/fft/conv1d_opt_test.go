//go:build amd64

package fft

import "testing"

func TestConv1DK3S1Identity(t *testing.T) {
	// kernel=[0,1,0] is identity
	inLen := 8
	input := make([]float32, inLen)
	for i := range input {
		input[i] = float32(i + 1)
	}
	weight := []float32{0, 1, 0} // 1 in, 1 out, k=3
	output := make([]float32, inLen)

	Conv1DK3S1(output, input, weight, nil, 1, inLen, 1)

	for i, v := range output {
		if v != input[i] {
			t.Fatalf("out[%d]=%f want %f", i, v, input[i])
		}
	}
}

func TestConv1DK3S2Downsample(t *testing.T) {
	inLen := 8
	input := make([]float32, inLen)
	for i := range input {
		input[i] = 1 // all ones
	}
	weight := []float32{1, 1, 1} // sum kernel
	outLen := (inLen+2-3)/2 + 1
	output := make([]float32, outLen)

	Conv1DK3S2(output, input, weight, nil, 1, inLen, 1)

	// With all-ones input and sum kernel, boundary elements have 2, middle have 3
	for i, v := range output {
		if v < 2 || v > 3 {
			t.Fatalf("out[%d]=%f want 2 or 3", i, v)
		}
	}
}

func BenchmarkConv1DK3S1_80x480(b *testing.B) {
	inCh := 80
	inLen := 480
	outCh := 384
	input := make([]float32, inCh*inLen)
	weight := make([]float32, outCh*inCh*3)
	output := make([]float32, outCh*inLen)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Conv1DK3S1(output, input, weight, nil, inCh, inLen, outCh)
	}
}
