package tensor

import (
	"math"
	"testing"
)

func TestConv1DIdentityKernel(t *testing.T) {
	// Single channel, kernel=[0,1,0], stride=1, padding=1 → identity
	input := [][]float32{{1, 2, 3, 4, 5}}
	weight := [][][]float32{{{0, 1, 0}}}
	out := Conv1D(input, weight, nil, 1, 1)
	if out == nil || len(out) != 1 || len(out[0]) != 5 {
		t.Fatalf("unexpected output shape: %v", out)
	}
	for i, v := range out[0] {
		if math.Abs(float64(v-input[0][i])) > 1e-6 {
			t.Fatalf("out[%d]=%f want %f", i, v, input[0][i])
		}
	}
}

func TestConv1DStride2(t *testing.T) {
	// kernel=[1,1,1], stride=2, padding=0 → sum of 3 consecutive, every other
	input := [][]float32{{1, 2, 3, 4, 5, 6}}
	weight := [][][]float32{{{1, 1, 1}}}
	out := Conv1D(input, weight, nil, 2, 0)
	// outLength = (6 - 3) / 2 + 1 = 2
	if out == nil || len(out[0]) != 2 {
		t.Fatalf("output length=%d want 2", len(out[0]))
	}
	if out[0][0] != 6 { // 1+2+3
		t.Fatalf("out[0]=%f want 6", out[0][0])
	}
	if out[0][1] != 12 { // 3+4+5
		t.Fatalf("out[1]=%f want 12", out[0][1])
	}
}

func TestConv1DWithBias(t *testing.T) {
	input := [][]float32{{1, 1, 1}}
	weight := [][][]float32{{{1, 1, 1}}}
	bias := []float32{10}
	out := Conv1D(input, weight, bias, 1, 1)
	// Middle element: 1+1+1+10 = 13
	if out[0][1] != 13 {
		t.Fatalf("out[1]=%f want 13", out[0][1])
	}
}

func TestConv1DFlat(t *testing.T) {
	// Same as identity test but flat
	input := []float32{1, 2, 3, 4, 5}
	weight := []float32{0, 1, 0}
	output := make([]float32, 5)
	Conv1DFlat(output, input, weight, nil, 1, 5, 1, 3, 1, 1)
	for i, v := range output {
		if math.Abs(float64(v-input[i])) > 1e-6 {
			t.Fatalf("flat out[%d]=%f want %f", i, v, input[i])
		}
	}
}

func TestConv1DMultiChannel(t *testing.T) {
	// 2 input channels, 1 output channel, kernel=1
	input := [][]float32{{1, 2}, {3, 4}}
	weight := [][][]float32{{{1}, {2}}} // out[j] = 1*in0[j] + 2*in1[j]
	out := Conv1D(input, weight, nil, 1, 0)
	if out == nil || len(out[0]) != 2 {
		t.Fatalf("unexpected shape")
	}
	// out[0] = 1*1 + 2*3 = 7, out[1] = 1*2 + 2*4 = 10
	if out[0][0] != 7 || out[0][1] != 10 {
		t.Fatalf("out=%v want [7,10]", out[0])
	}
}
