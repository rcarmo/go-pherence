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

// DotI8F32x4 computes four dot(int8(q), x[row]) outputs for contiguous rows
// separated by stride float32 elements. It is intended for small GGUF expert
// batches where the same raw quantized row is reused across selected positions.
func DotI8F32x4(q []byte, x []float32, stride int) (float32, float32, float32, float32, bool) {
	if len(q) == 0 || stride < len(q) || len(x) < 3*stride+len(q) {
		return 0, 0, 0, 0, false
	}
	q = q[:len(q)]
	s0, s1, s2, s3 := dotI8F32x4(q, x, stride)
	return s0, s1, s2, s3, true
}

func dotI8F32Scalar(q []byte, x []float32) float32 {
	var sum float32
	for i, b := range q {
		sum += float32(int8(b)) * x[i]
	}
	return sum
}

func dotI8F32x4Scalar(q []byte, x []float32, stride int) (float32, float32, float32, float32) {
	var s0, s1, s2, s3 float32
	x0 := x[:len(q)]
	x1 := x[stride : stride+len(q)]
	x2 := x[2*stride : 2*stride+len(q)]
	x3 := x[3*stride : 3*stride+len(q)]
	for i, b := range q {
		qv := float32(int8(b))
		s0 += qv * x0[i]
		s1 += qv * x1[i]
		s2 += qv * x2[i]
		s3 += qv * x3[i]
	}
	return s0, s1, s2, s3
}
