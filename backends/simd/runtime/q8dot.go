package simd

// DotI8F32 computes dot(int8(q), x) treating q bytes as signed int8 values.
// The optimized amd64 path requires len(q)==len(x) and a multiple of 8; other
// cases fall back to scalar validation/mathy for portability.
func DotI8F32(q []byte, x []float32) (float32, bool) {
	if len(q) == 0 || len(x) < len(q) {
		return 0, false
	}
	q = q[:len(q)]
	x = x[:len(q)]
	return dotI8F32(q, x), true
}

func dotI8F32Scalar(q []byte, x []float32) float32 {
	var sum float32
	for i, b := range q {
		sum += float32(int8(b)) * x[i]
	}
	return sum
}
