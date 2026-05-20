package simd

import "testing"

func TestCheckedBF16Entrypoints(t *testing.T) {
	x := BF16FromF32Slice([]float32{1, 2, 3, 4})
	y := BF16FromF32Slice([]float32{5, 6, 7, 8})
	yf := []float32{5, 6, 7, 8}
	if got, ok := BF16DotChecked(x, y); !ok || got == 0 {
		t.Fatalf("BF16DotChecked got=%g ok=%v", got, ok)
	}
	if got, ok := BF16DotF32Checked(x, yf); !ok || got == 0 {
		t.Fatalf("BF16DotF32Checked got=%g ok=%v", got, ok)
	}
	if !BF16RMSNormTo(x, y, 1e-6) {
		t.Fatal("BF16RMSNormTo rejected valid input")
	}
	out := make([]uint16, len(x)+1)
	out[len(out)-1] = 0x1234
	if !BF16VecAddTo(out[:len(x)], x, y) {
		t.Fatal("BF16VecAddTo rejected valid input")
	}
	if out[len(out)-1] != 0x1234 {
		t.Fatalf("BF16VecAddTo mutated tail: %v", out)
	}
	gemvOut := make([]uint16, 2)
	w := []float32{1, 0, 0, 1, 2, 0, 0, 2}
	if !BF16GemvNTTo(gemvOut, x, w, 4, 2) {
		t.Fatal("BF16GemvNTTo rejected valid input")
	}
}

func TestCheckedBF16EntrypointsRejectMalformedInputs(t *testing.T) {
	x := BF16FromF32Slice([]float32{1, 2, 3, 4})
	y := BF16FromF32Slice([]float32{5, 6, 7, 8})
	yf := []float32{5, 6, 7, 8}
	if _, ok := BF16DotChecked(nil, y); ok {
		t.Fatal("BF16DotChecked accepted nil x")
	}
	if _, ok := BF16DotChecked(x, y[:3]); ok {
		t.Fatal("BF16DotChecked accepted short y")
	}
	if _, ok := BF16DotF32Checked(nil, yf); ok {
		t.Fatal("BF16DotF32Checked accepted nil x")
	}
	if _, ok := BF16DotF32Checked(x, yf[:3]); ok {
		t.Fatal("BF16DotF32Checked accepted short y")
	}
	if BF16RMSNormTo(nil, y, 1e-6) || BF16RMSNormTo(x, y[:3], 1e-6) {
		t.Fatal("BF16RMSNormTo accepted malformed input")
	}
	if BF16VecAddTo(nil, x, y) || BF16VecAddTo(make([]uint16, 5), x, y) || BF16VecAddTo(make([]uint16, 4), x, y[:3]) {
		t.Fatal("BF16VecAddTo accepted malformed input")
	}
	if BF16GemvNTTo(make([]uint16, 1), x, yf, 4, 2) || BF16GemvNTTo(make([]uint16, 2), x[:3], yf, 4, 2) || BF16GemvNTTo(make([]uint16, 2), x, yf, 4, 2) {
		t.Fatal("BF16GemvNTTo accepted malformed input")
	}
}
