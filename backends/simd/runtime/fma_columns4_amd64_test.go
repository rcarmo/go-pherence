//go:build amd64

package simd

import (
	"math"
	"runtime"
	"testing"
)

func fmaColumns4TestControl(dst, x, weight []float32, mask uint32) bool

func TestFMAColumns4MXCSRAndIEEE(t *testing.T) {
	if !HasAffineF32Asm() {
		t.Skip("AVX2/FMA unavailable")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for _, mask := range []uint32{0x40, 0x8000, 0x2000, 0x4000, 0x6000} {
		d := []float32{9, 8, 7, 6, 5, 4, 3, 2}
		before := append([]float32(nil), d...)
		if fmaColumns4TestControl(d, []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4, 5, 6, 7, 8}, mask) {
			t.Fatal("MXCSR accepted", mask)
		}
		for i := range d {
			if d[i] != before[i] {
				t.Fatal("MXCSR write", mask, i)
			}
		}
	}
	const cols, k = 9, 5
	d := make([]float32, 4*cols)
	x := make([]float32, cols*k)
	w := make([]float32, 4*k)
	values := []float32{0, math.Float32frombits(0x80000000), math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, math.MaxFloat32, -math.MaxFloat32, 1, 1e-20, 1e20}
	for row := 0; row < k; row++ {
		copy(x[row*cols:], values)
	}
	for i := range w {
		w[i] = []float32{1, -1, .5, -.5, 2}[i%k]
	}
	want := make([]float32, len(d))
	fmaColumns4Scalar(want, x, w)
	if !fmaColumns4TestControl(d, x, w, 0) {
		t.Fatal("default")
	}
	for i := range d {
		if math.Float32bits(d[i]) != math.Float32bits(want[i]) {
			t.Fatal(i, d[i], want[i])
		}
	}
}
