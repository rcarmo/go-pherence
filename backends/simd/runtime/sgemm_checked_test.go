package simd

import (
	"math"
	"testing"
)

func TestSgemmNTTo(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{1, 0, 0, 1, 1, 1}
	c := []float32{10, 0, 0, 0, 123}
	if !SgemmNTTo(c[:4], a, b, 2, 2, 3, 1, 3, 3, 2) {
		t.Fatal("SgemmNTTo returned false")
	}
	want := []float32{11, 6, 4, 15}
	for i := range want {
		if c[i] != want[i] {
			t.Fatalf("c[%d]=%g want %g", i, c[i], want[i])
		}
	}
	if c[4] != 123 {
		t.Fatal("SgemmNTTo mutated tail")
	}
}

func TestSgemmNNTo(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{1, 2, 0, -1, 1, 0}
	c := []float32{0, 0, 0, 0, 123}
	if !SgemmNNTo(c[:4], a, b, 2, 2, 3, 1, 3, 2, 2) {
		t.Fatal("SgemmNNTo returned false")
	}
	want := []float32{4, 0, 10, 3}
	for i := range want {
		if c[i] != want[i] {
			t.Fatalf("c[%d]=%g want %g", i, c[i], want[i])
		}
	}
	if c[4] != 123 {
		t.Fatal("SgemmNNTo mutated tail")
	}
}

func TestSgemmNNToTwoRowTileAndTails(t *testing.T) {
	for _, tc := range []struct {
		m, n, k int
		alpha   float32
	}{
		{m: 2, n: 32, k: 7, alpha: 0.75},
		{m: 2, n: 41, k: 9, alpha: -0.5},
		{m: 3, n: 41, k: 9, alpha: 1.25},
		{m: 4, n: 65, k: 17, alpha: -0.75},
		{m: 5, n: 31, k: 6, alpha: 0.5},
	} {
		lda := tc.k + 3
		ldb := tc.n + 5
		ldc := tc.n + 7
		a := randFloats(tc.m*lda, int64(1000+tc.m*13+tc.n))
		b := randFloats(tc.k*ldb, int64(2000+tc.k*17+tc.n))
		want := randFloats(tc.m*ldc, int64(3000+tc.m*19+tc.k))
		got := append([]float32(nil), want...)

		for i := 0; i < tc.m; i++ {
			for j := 0; j < tc.n; j++ {
				sum := float32(0)
				for p := 0; p < tc.k; p++ {
					sum += a[i*lda+p] * b[p*ldb+j]
				}
				want[i*ldc+j] += tc.alpha * sum
			}
		}

		if !SgemmNNTo(got, a, b, tc.m, tc.n, tc.k, tc.alpha, lda, ldb, ldc) {
			t.Fatalf("SgemmNNTo rejected shape %+v", tc)
		}
		for i := range want {
			if diff := math.Abs(float64(got[i] - want[i])); diff > 5e-5 {
				t.Fatalf("shape=%+v index=%d got=%g want=%g diff=%g", tc, i, got[i], want[i], diff)
			}
		}
	}
}

func TestSgemmToRejectsMalformedInputs(t *testing.T) {
	a := []float32{1}
	b := []float32{1}
	c := []float32{1}
	if SgemmNTTo(c, a, b, 0, 1, 1, 1, 1, 1, 1) || SgemmNNTo(c, a, b, 1, 0, 1, 1, 1, 1, 1) {
		t.Fatal("accepted zero dimensions")
	}
	if SgemmNTTo(c, a, b, 1, 1, 2, 1, 1, 2, 1) || SgemmNNTo(c, a, b, 1, 1, 2, 1, 1, 1, 1) {
		t.Fatal("accepted short leading dimension")
	}
	if SgemmNTTo(c, a, b, 1, 1, 2, 1, 2, 2, 1) || SgemmNNTo(c, a, b, 1, 1, 2, 1, 2, 1, 1) {
		t.Fatal("accepted short backing slices")
	}
	maxInt := int(^uint(0) >> 1)
	if SgemmNTTo(c, a, b, maxInt/2+1, 1, 2, 1, 2, 2, 1) || SgemmNNTo(c, a, b, maxInt/2+1, 1, 2, 1, 2, 1, 1) {
		t.Fatal("accepted overflowing dimensions")
	}
	if SgemmNTTo(c, a, b, 2, 1, 1, 1, maxInt, 1, 1) || SgemmNNTo(c, a, b, 2, 1, 1, 1, maxInt, 1, 1) {
		t.Fatal("accepted overflowing A stride")
	}
	if SgemmNTTo(c, a, b, 1, 2, 1, 1, 1, maxInt, 1) || SgemmNNTo(c, a, b, 1, 1, 2, 1, 1, maxInt, 1) {
		t.Fatal("accepted overflowing B stride")
	}
	if SgemmNTTo(c, a, b, 2, 1, 1, 1, 1, 1, maxInt) || SgemmNNTo(c, a, b, 2, 1, 1, 1, 1, 1, maxInt) {
		t.Fatal("accepted overflowing C stride")
	}
}
