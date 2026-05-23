//go:build riscv64

package simd

import "testing"

func TestRVVRawSdotAsm(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -2, 0.5, 1.5, 2.5, 3.5, 4.5}
	y := []float32{0.5, -1, 2, 0.25, 3, -0.5, 1, 2, -3, 4, 5, -6, 7, -8, 9, -10, 11}
	got := sdotAsm(x, y)
	want := sdotScalar(x, y)
	if got != want {
		t.Fatalf("sdotAsm=%g want %g", got, want)
	}
}

func TestRVVRawSaxpyAsm(t *testing.T) {
	x := []float32{1, 2, 3, 4, 5, -1, -2, 0.5, 1.5}
	y := []float32{10, 20, 30, 40, 50, 1, 2, 3, 4}
	want := append([]float32(nil), y...)
	saxpyScalar(2.5, x, want)
	saxpyAsm(2.5, x, y)
	for i := range y {
		if y[i] != want[i] {
			t.Fatalf("y[%d]=%g want %g", i, y[i], want[i])
		}
	}
}
