package simd

// VecMulAddTo computes dst[i] = dst[i] + float32(a[i]*b[i]) in one pass.
// Multiplication and addition round separately to float32; this is NOT FMA.
// All slices must have equal non-zero lengths. Exact dst/a or dst/b aliasing
// is allowed, but partial output/input overlap is rejected before mutation.
// The read-only inputs may overlap each other. No scratch or allocation is used.
func VecMulAddTo(dst, a, b []float32) bool {
	n := len(dst)
	if n == 0 || len(a) != n || len(b) != n {
		return false
	}
	if !expF32AliasOK(dst, a) || !expF32AliasOK(dst, b) {
		return false
	}
	vecMulAdd(dst, a, b)
	return true
}

func vecMulAddScalar(dst, a, b []float32) {
	for i := range dst {
		// Explicit conversion forbids contraction into a fused multiply-add.
		dst[i] += float32(a[i] * b[i])
	}
}
