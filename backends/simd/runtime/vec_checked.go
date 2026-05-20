package simd

// VecAddTo computes dst = a + b and reports malformed inputs.
func VecAddTo(dst, a, b []float32) bool {
	if len(dst) == 0 || len(a) < len(dst) || len(b) < len(dst) {
		return false
	}
	VecAdd(dst, a, b)
	return true
}

// VecMulTo computes dst = a * b and reports malformed inputs.
func VecMulTo(dst, a, b []float32) bool {
	if len(dst) == 0 || len(a) < len(dst) || len(b) < len(dst) {
		return false
	}
	VecMul(dst, a, b)
	return true
}

// VecScaleAddTo computes dst = a + scale*b and reports malformed inputs.
func VecScaleAddTo(dst, a, b []float32, scale float32) bool {
	if len(dst) == 0 || len(a) < len(dst) || len(b) < len(dst) {
		return false
	}
	VecScaleAdd(dst, a, b, scale)
	return true
}

// VecScaleTo computes dst = scale*a and reports malformed inputs.
func VecScaleTo(dst, a []float32, scale float32) bool {
	if len(dst) == 0 || len(a) < len(dst) {
		return false
	}
	VecScale(dst, a, scale)
	return true
}

// RMSNormTo normalizes x in place with weights and reports malformed inputs.
func RMSNormTo(x, w []float32, eps float32) bool {
	if len(x) == 0 || len(w) < len(x) {
		return false
	}
	RMSNorm(x, w, eps)
	return true
}

// RMSNormNoScaleTo normalizes x in place without weights and reports malformed inputs.
func RMSNormNoScaleTo(x []float32, eps float32) bool {
	if len(x) == 0 {
		return false
	}
	RMSNormNoScale(x, eps)
	return true
}
