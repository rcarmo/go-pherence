//go:build amd64

package simd

import (
	"math"
	"runtime"
	"testing"
)

func fmaMatrixTestControl(dst, a, b []float32, m, n, k int, mask uint32) bool
func TestFMAMatrixMXCSRAndIEEE(t *testing.T) {
	if !HasAffineF32Asm() {
		t.Skip("AVX2/FMA unavailable")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for _, mask := range []uint32{0x40, 0x8000, 0x2000, 0x4000, 0x6000} {
		dst := []float32{7, 8}
		if fmaMatrixTestControl(dst, []float32{1}, []float32{2, 3}, 1, 2, 1, mask) || dst[0] != 7 || dst[1] != 8 {
			t.Fatal("control", mask)
		}
	}
	vals := []float32{0, math.Float32frombits(0x80000000), math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32, math.MaxFloat32, -math.MaxFloat32, 1, -1, 1e-20}
	a := []float32{1, 1, 1, 1, 1, 1}
	b := make([]float32, 3*len(vals))
	for i := 0; i < 3; i++ {
		copy(b[i*len(vals):], vals)
	}
	dst, want := make([]float32, 2*len(vals)), make([]float32, 2*len(vals))
	fmaMatrixScalar(want, a, b, 2, len(vals), 3)
	if !fmaMatrixTestControl(dst, a, b, 2, len(vals), 3, 0) {
		t.Fatal("default")
	}
	for i, v := range dst {
		if math.Float32bits(v) != math.Float32bits(want[i]) {
			t.Fatal("IEEE", i, v, want[i])
		}
	}
}
