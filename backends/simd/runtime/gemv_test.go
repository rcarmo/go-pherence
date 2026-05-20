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

func TestGemmRowsAndCols(t *testing.T) {
	xRows := []float32{1, -2, 3, 0.5, -1, 2}
	wRows := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	outRows := make([]float32, 2*2+1)
	outRows[len(outRows)-1] = 123
	if !GemmRows(outRows, xRows, wRows, 2, 2, 3) {
		t.Fatal("GemmRows returned false")
	}
	wantRows := []float32{6, 4, 4.5, 3}
	for i := range wantRows {
		if outRows[i] != wantRows[i] {
			t.Fatalf("GemmRows out[%d]=%g want %g", i, outRows[i], wantRows[i])
		}
	}
	if outRows[len(outRows)-1] != 123 {
		t.Fatal("GemmRows mutated tail")
	}
	if GemmRows(outRows[:3], xRows, wRows, 2, 2, 3) {
		t.Fatal("GemmRows accepted short output")
	}

	xCols := []float32{1, -2, 0.5, -1}
	wCols := []float32{
		1, 2, 3,
		-1, 0.5, 2,
	}
	outCols := make([]float32, 2*3+1)
	outCols[len(outCols)-1] = 123
	if !GemmCols(outCols, xCols, wCols, 2, 2, 3) {
		t.Fatal("GemmCols returned false")
	}
	wantCols := []float32{3, 1, -1, 1.5, 0.5, -0.5}
	for i := range wantCols {
		if outCols[i] != wantCols[i] {
			t.Fatalf("GemmCols out[%d]=%g want %g", i, outCols[i], wantCols[i])
		}
	}
	if outCols[len(outCols)-1] != 123 {
		t.Fatal("GemmCols mutated tail")
	}
	if GemmCols(outCols[:5], xCols, wCols, 2, 2, 3) {
		t.Fatal("GemmCols accepted short output")
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
