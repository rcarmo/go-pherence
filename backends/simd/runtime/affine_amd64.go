//go:build amd64

package simd

import "golang.org/x/sys/cpu"

// HasAffineF32Asm reports runtime CPU+OS support, independently of mutable
// package feature toggles used by older kernels. No AVX executes without it.
func HasAffineF32Asm() bool { return cpu.X86.HasAVX2 && cpu.X86.HasFMA }

func affineF32InPlace(x []float32, scale, shift float32) bool {
	if HasAffineF32Asm() {
		return affineF32Asm(x, scale, shift)
	}
	affineF32Scalar(x, scale, shift)
	return true
}

//go:noescape
func affineF32Asm(x []float32, scale, shift float32) bool
