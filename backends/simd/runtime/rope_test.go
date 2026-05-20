package simd

import "testing"

func TestApplyRoPECheckedRuntimeWrappers(t *testing.T) {
	x := []float32{1, 2, 3, 4, 99}
	if !ApplyRoPEPartialTo(x, []float32{0, 1}, 0, 1, 4, 1) {
		t.Fatal("ApplyRoPEPartialTo returned false for valid input")
	}
	want := []float32{-2, 1, 3, 4, 99}
	for i := range want {
		if x[i] != want[i] {
			t.Fatalf("x[%d]=%g want %g (all=%v)", i, x[i], want[i], x)
		}
	}
	if ApplyRoPEPartialTo(x, []float32{0}, 0, 1, 4, 1) {
		t.Fatal("ApplyRoPEPartialTo accepted short freqs")
	}
	if ApplyRoPEPartialTo(x[:3], []float32{0, 1}, 0, 1, 4, 1) {
		t.Fatal("ApplyRoPEPartialTo accepted short x")
	}
	if ApplyRoPETo(x[:4], []float32{0, 1, 1, 0}, 0, 1, 4) == false {
		t.Fatal("ApplyRoPETo rejected valid full RoPE")
	}
}

func TestApplyRoPEPartialRuntimeWrapper(t *testing.T) {
	x := []float32{1, 2, 3, 4, 99}
	ApplyRoPEPartial(x, []float32{0, 1}, 0, 1, 4, 1)
	want := []float32{-2, 1, 3, 4, 99}
	for i := range want {
		if x[i] != want[i] {
			t.Fatalf("x[%d]=%g want %g (all=%v)", i, x[i], want[i], x)
		}
	}
}
