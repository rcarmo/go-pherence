package simd

// BF16DotF32Checked computes dot(BF16,F32) and reports malformed inputs.
func BF16DotF32Checked(x []uint16, y []float32) (float32, bool) {
	if len(x) == 0 || len(y) < len(x) {
		return 0, false
	}
	return BF16DotF32(x, y), true
}

// BF16DotChecked computes dot(BF16,BF16) and reports malformed inputs.
func BF16DotChecked(x, y []uint16) (float32, bool) {
	if len(x) == 0 || len(y) < len(x) {
		return 0, false
	}
	return BF16Dot(x, y), true
}

// BF16RMSNormTo computes BF16 RMSNorm in-place and reports malformed inputs.
func BF16RMSNormTo(x, w []uint16, eps float32) bool {
	if len(x) == 0 || len(w) < len(x) {
		return false
	}
	BF16RMSNorm(x, w, eps)
	return true
}

// BF16VecAddTo computes BF16 vector add and reports malformed inputs.
func BF16VecAddTo(dst, a, b []uint16) bool {
	if len(dst) == 0 || len(a) < len(dst) || len(b) < len(dst) {
		return false
	}
	BF16VecAdd(dst, a, b)
	return true
}

// BF16GemvNTTo computes mixed BF16/F32 GEMV and reports malformed inputs.
func BF16GemvNTTo(out []uint16, x []uint16, w []float32, inDim, outDim int) bool {
	weightLen, ok := checkedMulInt(inDim, outDim)
	if inDim <= 0 || outDim <= 0 || !ok || len(out) < outDim || len(x) < inDim || len(w) < weightLen {
		return false
	}
	BF16GemvNT(out, x, w, inDim, outDim)
	return true
}
