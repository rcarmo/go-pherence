package activation

import "testing"

func TestCheckedActivationEntrypoints(t *testing.T) {
	a := []float32{-1, 0, 2}
	b := []float32{2, 3, 4}
	dst := make([]float32, len(a)+1)
	dst[len(dst)-1] = 123
	if !SiLUTo(dst[:len(a)], a) {
		t.Fatal("SiLUTo returned false for valid input")
	}
	if !SiLUMulTo(dst[:len(a)], a, b) {
		t.Fatal("SiLUMulTo returned false for valid input")
	}
	if !GELUTanhTo(dst[:len(a)], a) {
		t.Fatal("GELUTanhTo returned false for valid input")
	}
	got, ok := GELUTanhChecked(a)
	if !ok || len(got) != len(a) || got[1] != 0 {
		t.Fatalf("GELUTanhChecked got=%v ok=%v", got, ok)
	}
	if !GELUTanhMulTo(dst[:len(a)], a, b) {
		t.Fatal("GELUTanhMulTo returned false for valid input")
	}
	if dst[len(dst)-1] != 123 {
		t.Fatalf("checked activation mutated tail: %v", dst)
	}
	if SiLUTo(nil, a) || SiLUTo(make([]float32, 4), a) {
		t.Fatal("SiLUTo accepted malformed input")
	}
	if SiLUMulTo(nil, a, b) || SiLUMulTo(make([]float32, 4), a, b) || SiLUMulTo(make([]float32, 3), a, b[:2]) {
		t.Fatal("SiLUMulTo accepted malformed input")
	}
	if GELUTanhTo(nil, a) || GELUTanhTo(make([]float32, 4), a) {
		t.Fatal("GELUTanhTo accepted malformed input")
	}
	if got, ok := GELUTanhChecked(nil); ok || got != nil {
		t.Fatalf("GELUTanhChecked accepted nil: %v %v", got, ok)
	}
	if GELUTanhMulTo(nil, a, b) || GELUTanhMulTo(make([]float32, 4), a, b) || GELUTanhMulTo(make([]float32, 3), a, b[:2]) {
		t.Fatal("GELUTanhMulTo accepted malformed input")
	}
}
