//go:build amd64

package simd

import (
	"runtime"
	"testing"
)

// Test shim changes control and restores it in ONE NOSPLIT assembly call.
// No Go/runtime call executes in the temporary non-default FP environment.
func affineTestControl(x []float32, scale, shift float32, mask uint32) bool

func TestAffineF32RejectsNondefaultMXCSR(t *testing.T) {
	if !HasAffineF32Asm() {
		t.Skip("AVX2/FMA unavailable")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for _, mask := range []uint32{0x40, 0x8000, 0x2000, 0x4000, 0x6000} {
		x := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}
		if affineTestControl(x, 2, 3, mask) {
			t.Fatal("non-default FP controls accepted", mask)
		}
		for i, v := range x {
			if v != float32(i+1) {
				t.Fatal("rejected controls modified buffer")
			}
		}
	}
	x := []float32{1, 2}
	if !affineTestControl(x, 2, 3, 0) || x[0] != 5 || x[1] != 7 {
		t.Fatal("default controls rejected")
	}
}
