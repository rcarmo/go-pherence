//go:build amd64

package simd

import (
	"math"
	"runtime"
	"testing"
)

func fmaColumnsTestControl(dst, x, weight []float32, mask uint32) bool

func TestFMAColumnsMXCSRAndIEEE(t *testing.T) {
	if !HasAffineF32Asm() {
		t.Skip("AVX2/FMA unavailable")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for _, mask := range []uint32{0x40, 0x8000, 0x2000, 0x4000, 0x6000} {
		d := []float32{9, 8}
		if fmaColumnsTestControl(d, []float32{1, 2}, []float32{2}, mask) || d[0] != 9 || d[1] != 8 {
			t.Fatal("MXCSR", mask)
		}
	}
	d := make([]float32, 9)
	x := make([]float32, 9*5)
	w := []float32{1, 1, 1, 1, 1}
	vals := []float32{0, math.Float32frombits(0x80000000), math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, math.MaxFloat32, -math.MaxFloat32, 1, 1e-20, 1e20}
	for k := range w {
		copy(x[k*9:], vals)
	}
	want := make([]float32, 9)
	fmaColumnsScalar(want, x, w)
	if !fmaColumnsTestControl(d, x, w, 0) {
		t.Fatal("default")
	}
	for i := range d {
		if math.Float32bits(d[i]) != math.Float32bits(want[i]) {
			t.Fatal(i, d[i], want[i])
		}
	}
}
