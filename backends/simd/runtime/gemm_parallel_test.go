package simd

import (
	"math/rand"
	"testing"
)

func randFloats(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = r.Float32()*2 - 1
	}
	return out
}

func floatsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestGemmRowsParallelMatchesSerial checks the parallel batched NT GEMM is
// bit-identical to the serial GemmRows reference.
func TestGemmRowsParallelMatchesSerial(t *testing.T) {
	batch, rows, cols := 5, 640, 48
	x := randFloats(batch*cols, 1)
	w := randFloats(rows*cols, 2)
	want := make([]float32, batch*rows)
	got := make([]float32, batch*rows)
	if !GemmRows(want, x, w, batch, rows, cols) {
		t.Fatal("GemmRows failed")
	}
	if !GemmRowsParallel(got, x, w, batch, rows, cols) {
		t.Fatal("GemmRowsParallel failed")
	}
	if !floatsEqual(want, got) {
		t.Fatal("GemmRowsParallel result differs from GemmRows")
	}
}

// TestSgemmNNParallelMatchesSerial checks the column-parallel NN GEMM is
// bit-identical to a single SgemmNNTo call.
func TestSgemmNNParallelMatchesSerial(t *testing.T) {
	m, n, k := 5, 640, 48
	a := randFloats(m*k, 3)
	b := randFloats(k*n, 4)
	want := make([]float32, m*n)
	got := make([]float32, m*n)
	if !SgemmNNTo(want, a, b, m, n, k, 1.0, k, n, n) {
		t.Fatal("SgemmNNTo failed")
	}
	if !SgemmNNParallelTo(got, a, b, m, n, k, 1.0, k, n, n) {
		t.Fatal("SgemmNNParallelTo failed")
	}
	if !floatsEqual(want, got) {
		t.Fatal("SgemmNNParallelTo result differs from SgemmNNTo")
	}
}

func TestGemmParallelMalformedInputs(t *testing.T) {
	if GemmRowsParallel(nil, nil, nil, 0, 1, 1) {
		t.Fatal("GemmRowsParallel accepted bad dims")
	}
	if SgemmNNParallelTo(nil, nil, nil, 0, 1, 1, 1.0, 1, 1, 1) {
		t.Fatal("SgemmNNParallelTo accepted bad dims")
	}
}
