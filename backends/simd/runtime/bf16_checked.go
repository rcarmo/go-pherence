package simd

// BF16DotF32Checked computes dot(BF16,F32) and reports malformed inputs.
func BF16DotF32To(x []uint16, y []float32) (float32, bool) {
	return BF16DotF32Checked(x, y)
}

// BF16DotChecked computes dot(BF16,BF16) and reports malformed inputs.
func BF16DotTo(x, y []uint16) (float32, bool) {
	return BF16DotChecked(x, y)
}

// BF16RMSNormTo computes BF16 RMSNorm in-place and reports malformed inputs.
func BF16RMSNormTo(x, w []uint16, eps float32) bool {
	return BF16RMSNormChecked(x, w, eps)
}

// BF16VecAddTo computes BF16 vector add and reports malformed inputs.
func BF16VecAddTo(dst, a, b []uint16) bool {
	return BF16VecAddChecked(dst, a, b)
}

// BF16GemvNTTo computes mixed BF16/F32 GEMV and reports malformed inputs.
func BF16GemvNTTo(out []uint16, x []uint16, w []float32, inDim, outDim int) bool {
	return BF16GemvNTChecked(out, x, w, inDim, outDim)
}
