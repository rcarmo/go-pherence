package vector

import "testing"

func TestCheckedVectorEntrypoints(t *testing.T) {
	a := []float32{1, 2, 3}
	b := []float32{4, 5, 6}
	dst := make([]float32, 4)
	dst[3] = 123
	if !VecAddTo(dst[:3], a, b) || dst[0] != 5 {
		t.Fatalf("VecAddTo failed: %v", dst)
	}
	if !VecMulTo(dst[:3], a, b) || dst[0] != 4 {
		t.Fatalf("VecMulTo failed: %v", dst)
	}
	if !VecScaleAddTo(dst[:3], a, b, 0.5) || dst[0] != 3 {
		t.Fatalf("VecScaleAddTo failed: %v", dst)
	}
	if !VecScaleTo(dst[:3], a, 2) || dst[0] != 2 {
		t.Fatalf("VecScaleTo failed: %v", dst)
	}
	if dst[3] != 123 {
		t.Fatalf("checked vector mutated tail: %v", dst)
	}
	if VecAddTo(nil, a, b) || VecAddTo(make([]float32, 4), a, b) {
		t.Fatal("VecAddTo accepted malformed input")
	}
	if VecMulTo(nil, a, b) || VecMulTo(make([]float32, 4), a, b) {
		t.Fatal("VecMulTo accepted malformed input")
	}
	if VecScaleAddTo(nil, a, b, 1) || VecScaleAddTo(make([]float32, 4), a, b, 1) {
		t.Fatal("VecScaleAddTo accepted malformed input")
	}
	if VecScaleTo(nil, a, 1) || VecScaleTo(make([]float32, 4), a, 1) {
		t.Fatal("VecScaleTo accepted malformed input")
	}
}

func TestCheckedRMSNormEntrypoints(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	w := []float32{1, 1, 1, 1}
	if !RMSNormTo(x, w, 1e-6) {
		t.Fatal("RMSNormTo rejected valid input")
	}
	if !RMSNormNoScaleTo(x, 1e-6) {
		t.Fatal("RMSNormNoScaleTo rejected valid input")
	}
	if RMSNormTo(nil, w, 1e-6) || RMSNormTo(make([]float32, 4), w[:3], 1e-6) {
		t.Fatal("RMSNormTo accepted malformed input")
	}
	if RMSNormNoScaleTo(nil, 1e-6) {
		t.Fatal("RMSNormNoScaleTo accepted malformed input")
	}
}
