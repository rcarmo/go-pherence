package bf16

import "testing"

func TestCheckedEntrypoints(t *testing.T) {
	x := BF16FromF32Slice([]float32{1, 2, 3, 4})
	y := BF16FromF32Slice([]float32{5, 6, 7, 8})
	yf := []float32{5, 6, 7, 8}
	if _, ok := BF16DotChecked(x, y); !ok {
		t.Fatal("BF16DotChecked rejected valid input")
	}
	if _, ok := BF16DotF32Checked(x, yf); !ok {
		t.Fatal("BF16DotF32Checked rejected valid input")
	}
	if !BF16RMSNormChecked(x, y, 1e-6) {
		t.Fatal("BF16RMSNormChecked rejected valid input")
	}
	if !BF16VecAddChecked(make([]uint16, 4), x, y) {
		t.Fatal("BF16VecAddChecked rejected valid input")
	}
	if !BF16GemvNTChecked(make([]uint16, 2), x, []float32{1, 0, 0, 1, 2, 0, 0, 2}, 4, 2) {
		t.Fatal("BF16GemvNTChecked rejected valid input")
	}
}

func TestCheckedEntrypointsRejectMalformedInputs(t *testing.T) {
	x := BF16FromF32Slice([]float32{1, 2, 3, 4})
	y := BF16FromF32Slice([]float32{5, 6, 7, 8})
	yf := []float32{5, 6, 7, 8}
	if _, ok := BF16DotChecked(nil, y); ok {
		t.Fatal("BF16DotChecked accepted nil x")
	}
	if _, ok := BF16DotF32Checked(x, yf[:3]); ok {
		t.Fatal("BF16DotF32Checked accepted short y")
	}
	if BF16RMSNormChecked(x, y[:3], 1e-6) {
		t.Fatal("BF16RMSNormChecked accepted short w")
	}
	if BF16VecAddChecked(make([]uint16, 5), x, y) {
		t.Fatal("BF16VecAddChecked accepted short input")
	}
	if BF16GemvNTChecked(make([]uint16, 2), x, yf, 4, 2) {
		t.Fatal("BF16GemvNTChecked accepted short weights")
	}
}
