package simd

import "testing"

func TestGemvRows(t *testing.T) {
	x := []float32{1, -2, 3}
	w := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	out := []float32{0, 0, 123}
	if !GemvRows(out[:2], x, w, 2, 3) {
		t.Fatal("GemvRows returned false")
	}
	want := []float32{6, 4}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d]=%g want %g", i, out[i], want[i])
		}
	}
	if out[2] != 123 {
		t.Fatal("GemvRows mutated tail")
	}
	if GemvRows(out[:1], x, w, 2, 3) || GemvRows(out[:2], x[:2], w, 2, 3) || GemvRows(out[:2], x, w[:5], 2, 3) {
		t.Fatal("GemvRows accepted malformed input")
	}
}

func TestGemvCols(t *testing.T) {
	x := []float32{1, -2}
	w := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	out := []float32{0, 0, 0, 123}
	if !GemvCols(out[:3], x, w, 2, 3) {
		t.Fatal("GemvCols returned false")
	}
	want := []float32{3, 1, -1}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d]=%g want %g", i, out[i], want[i])
		}
	}
	if out[3] != 123 {
		t.Fatal("GemvCols mutated tail")
	}
	if GemvCols(out[:2], x, w, 2, 3) || GemvCols(out[:3], x[:1], w, 2, 3) || GemvCols(out[:3], x, w[:5], 2, 3) {
		t.Fatal("GemvCols accepted malformed input")
	}
}
