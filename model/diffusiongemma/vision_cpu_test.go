package diffusiongemma

import (
	"math"
	"testing"
)

func TestApplyVisionRoPE2DSeparatesXYChannels(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	if !applyVisionRoPE2D(x, 8, 1, 0, 10000) {
		t.Fatal("RoPE rejected valid head")
	}
	want := []float32{-1.984111, 1.959901, 2.462378, 4.019800, 5, 6, 7, 8}
	for i := range want {
		if math.Abs(float64(x[i]-want[i])) > 1e-5 {
			t.Fatalf("x[%d]=%.6f want %.6f full=%v", i, x[i], want[i], x)
		}
	}
}

func TestRunVisionLayerF32RejectsBadShape(t *testing.T) {
	layer := tinyVisionLayerF32(2, 1, 2, 3)
	if err := RunVisionLayerF32([]float32{1, 2, 3}, 2, 2, 1, 2, layer); err == nil {
		t.Fatal("expected bad hidden length error")
	}
}

func TestRunVisionLayerF32Deterministic(t *testing.T) {
	layer := tinyVisionLayerF32(2, 1, 2, 3)
	hidden := []float32{1, -0.5, 0.25, 0.75}
	if err := RunVisionLayerF32(hidden, 2, 2, 1, 2, layer); err != nil {
		t.Fatal(err)
	}
	want := []float32{3.245795, -2.071715, 0.388404, 3.382011}
	if len(hidden) != len(want) {
		t.Fatalf("len=%d want %d", len(hidden), len(want))
	}
	for i := range want {
		if math.Abs(float64(hidden[i]-want[i])) > 1e-5 {
			t.Fatalf("hidden[%d]=%.6f want %.6f full=%v", i, hidden[i], want[i], hidden)
		}
	}
}

func tinyVisionLayerF32(hiddenSize, heads, headDim, intermediate int) VisionLayerF32 {
	id := func(n int) []float32 {
		m := make([]float32, n*n)
		for i := 0; i < n; i++ {
			m[i*n+i] = 1
		}
		return m
	}
	vec := func(n int, v float32) []float32 {
		out := make([]float32, n)
		for i := range out {
			out[i] = v
		}
		return out
	}
	gate := []float32{0.5, -0.25, 0.75, 0.5, -0.5, 0.25}
	up := []float32{1, 0.25, -0.5, 0.5, 0.25, 1}
	down := []float32{0.5, -0.25, 0.75, -0.5, 0.5, 0.25}
	return VisionLayerF32{
		InputLayerNorm:         vec(hiddenSize, 1),
		PostAttentionLayerNorm: vec(hiddenSize, 1),
		PreFFNLayerNorm:        vec(hiddenSize, 1),
		PostFFNLayerNorm:       vec(hiddenSize, 1),
		QProj:                  id(hiddenSize),
		KProj:                  id(hiddenSize),
		VProj:                  id(hiddenSize),
		OProj:                  id(hiddenSize),
		QNorm:                  vec(headDim, 1),
		KNorm:                  vec(headDim, 1),
		MLPGateProj:            gate,
		MLPUpProj:              up,
		MLPDownProj:            down,
		MLPIntermediate:        intermediate,
	}
}
