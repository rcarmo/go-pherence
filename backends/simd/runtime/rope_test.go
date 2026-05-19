package simd

import "testing"

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
