package whisper

import (
	"math"
	"testing"
)

func TestLinearRowBlockUsesSIMDOracleMatchesScalar(t *testing.T) {
	oldBlockM := blockM
	blockM = 3
	defer func() { blockM = oldBlockM }()

	m, inDim, outDim := 7, 11, 13
	x := make([]float32, m*inDim)
	w := make([]float32, outDim*inDim)
	bias := make([]float32, outDim)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.03125
	}
	for i := range w {
		w[i] = float32((i%19)-9) * 0.015625
	}
	for i := range bias {
		bias[i] = float32((i%5)-2) * 0.125
	}

	got := make([]float32, m*outDim)
	want := linearReference(x, w, bias, m, inDim, outDim)
	linearRowBlock(got, x, w, bias, 0, m, inDim, outDim)
	assertCloseFloat32(t, got, want, 1e-5)
}

func TestLinearRowBlockShortBiasMatchesLegacySemantics(t *testing.T) {
	oldBlockM := blockM
	blockM = 2
	defer func() { blockM = oldBlockM }()

	m, inDim, outDim := 4, 5, 6
	x := []float32{
		1, -2, 3, -4, 5,
		-1, 2, -3, 4, -5,
		0.5, 0.25, -0.75, 1.25, -1.5,
		2, 0, -2, 1, -1,
	}
	w := make([]float32, outDim*inDim)
	for i := range w {
		w[i] = float32((i%7)-3) * 0.2
	}
	bias := []float32{0.75, -0.5, 0.25}

	got := make([]float32, m*outDim)
	want := linearReference(x, w, bias, m, inDim, outDim)
	linearRowBlock(got, x, w, bias, 0, m, inDim, outDim)
	assertCloseFloat32(t, got, want, 1e-5)
}

func linearReference(x, w, bias []float32, m, inDim, outDim int) []float32 {
	out := make([]float32, m*outDim)
	for i := 0; i < m; i++ {
		for o := 0; o < outDim; o++ {
			var sum float32
			for d := 0; d < inDim; d++ {
				sum += x[i*inDim+d] * w[o*inDim+d]
			}
			if o < len(bias) {
				sum += bias[o]
			}
			out[i*outDim+o] = sum
		}
	}
	return out
}

func assertCloseFloat32(t *testing.T, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range got {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > tol {
			t.Fatalf("output[%d] mismatch: got %.8f want %.8f diff %.8f", i, got[i], want[i], diff)
		}
	}
}
