//go:build amd64 || arm64

package simd

// Sdot computes the dot product of two float32 slices using assembly when
// available, falling back to scalar code otherwise.
func Sdot(x, y []float32) float32 {
	// Assembly kernels assume equal lengths. Preserve scalar fallback semantics
	// for defensive callers that pass mismatched slices.
	if len(x) == len(y) && HasDotAsm {
		return sdotAsm(x, y)
	}
	return sdotScalar(x, y)
}
