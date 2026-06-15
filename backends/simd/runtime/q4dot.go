package simd

// DotU4F32LowAndSum computes dot(low_nibble(q[i]), x[i]) and sum(x[i]).
func DotU4F32LowAndSum(q []byte, x []float32) (float32, float32, bool) {
	if len(q) == 0 || len(x) < len(q) {
		return 0, 0, false
	}
	q = q[:len(q)]
	x = x[:len(q)]
	dot, sum := dotU4F32LowAndSum(q, x)
	return dot, sum, true
}

// DotU4F32HighAndSum computes dot(high_nibble(q[i]), x[i]) and sum(x[i]).
func DotU4F32HighAndSum(q []byte, x []float32) (float32, float32, bool) {
	if len(q) == 0 || len(x) < len(q) {
		return 0, 0, false
	}
	q = q[:len(q)]
	x = x[:len(q)]
	dot, sum := dotU4F32HighAndSum(q, x)
	return dot, sum, true
}

func dotU4F32LowAndSumScalar(q []byte, x []float32) (float32, float32) {
	var dot, sum float32
	for i, b := range q {
		xv := x[i]
		dot += float32(b&0x0F) * xv
		sum += xv
	}
	return dot, sum
}

func dotU4F32HighAndSumScalar(q []byte, x []float32) (float32, float32) {
	var dot, sum float32
	for i, b := range q {
		xv := x[i]
		dot += float32(b>>4) * xv
		sum += xv
	}
	return dot, sum
}
