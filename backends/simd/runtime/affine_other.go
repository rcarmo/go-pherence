//go:build !amd64

package simd

func HasAffineF32Asm() bool { return false }
func affineF32InPlace(x []float32, scale, shift float32) bool {
	affineF32Scalar(x, scale, shift)
	return true
}
