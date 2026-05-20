package activation

import "testing"

func TestAddBiasRowsTo(t *testing.T) {
	dst := []float32{1, 2, 3, 4, 5, 6, 123}
	bias := []float32{0.5, -1, 2}
	if !AddBiasRowsTo(dst[:6], bias, 2, 3) {
		t.Fatal("AddBiasRowsTo returned false")
	}
	want := []float32{1.5, 1, 5, 4.5, 4, 8}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("dst[%d]=%g want %g", i, dst[i], want[i])
		}
	}
	if dst[6] != 123 {
		t.Fatal("AddBiasRowsTo mutated tail")
	}
}

func TestAddBiasRowsToRejectsMalformedInputs(t *testing.T) {
	dst := make([]float32, 6)
	bias := make([]float32, 3)
	if AddBiasRowsTo(dst[:5], bias, 2, 3) {
		t.Fatal("accepted short dst")
	}
	if AddBiasRowsTo(dst, bias[:2], 2, 3) {
		t.Fatal("accepted short bias")
	}
	if AddBiasRowsTo(dst, bias, 0, 3) || AddBiasRowsTo(dst, bias, 2, 0) {
		t.Fatal("accepted zero dimensions")
	}
	maxInt := int(^uint(0) >> 1)
	if AddBiasRowsTo(dst, bias, maxInt/2+1, 3) {
		t.Fatal("accepted overflowing dimensions")
	}
}
