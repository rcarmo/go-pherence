package simd

import "testing"

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
