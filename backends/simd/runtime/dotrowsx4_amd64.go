//go:build amd64

package simd

const HasDotRowsx4SIMD = true

//go:noescape
func dotRowsx4Asm(w []float32, x []float32, cols int) (dot0 float32, dot1 float32, dot2 float32, dot3 float32)

func dotRowsx4(w []float32, x []float32, cols int) (float32, float32, float32, float32) {
	if !HasDotAsm {
		return dotRowsx4Scalar(w, x, cols)
	}
	return dotRowsx4Asm(w, x, cols)
}
