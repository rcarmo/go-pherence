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

// DotU4F32LowAndSumx4 computes four dot(low_nibble(q[i]), x[row][i])
// and sum(x[row][i]) outputs for contiguous rows separated by stride.
func DotU4F32LowAndSumx4(q []byte, x []float32, stride int) (float32, float32, float32, float32, float32, float32, float32, float32, bool) {
	if len(q) == 0 || stride < len(q) || len(x) < 3*stride+len(q) {
		return 0, 0, 0, 0, 0, 0, 0, 0, false
	}
	d0, s0, d1, s1, d2, s2, d3, s3 := dotU4F32LowAndSumx4(q, x, stride)
	return d0, s0, d1, s1, d2, s2, d3, s3, true
}

// DotU4F32HighAndSumx4 computes four dot(high_nibble(q[i]), x[row][i])
// and sum(x[row][i]) outputs for contiguous rows separated by stride.
func DotU4F32HighAndSumx4(q []byte, x []float32, stride int) (float32, float32, float32, float32, float32, float32, float32, float32, bool) {
	if len(q) == 0 || stride < len(q) || len(x) < 3*stride+len(q) {
		return 0, 0, 0, 0, 0, 0, 0, 0, false
	}
	d0, s0, d1, s1, d2, s2, d3, s3 := dotU4F32HighAndSumx4(q, x, stride)
	return d0, s0, d1, s1, d2, s2, d3, s3, true
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

func dotU4F32LowAndSumx4Scalar(q []byte, x []float32, stride int) (float32, float32, float32, float32, float32, float32, float32, float32) {
	var d0, s0, d1, s1, d2, s2, d3, s3 float32
	x0 := x[:len(q)]
	x1 := x[stride : stride+len(q)]
	x2 := x[2*stride : 2*stride+len(q)]
	x3 := x[3*stride : 3*stride+len(q)]
	for i, b := range q {
		qv := float32(b & 0x0F)
		d0 += qv * x0[i]
		s0 += x0[i]
		d1 += qv * x1[i]
		s1 += x1[i]
		d2 += qv * x2[i]
		s2 += x2[i]
		d3 += qv * x3[i]
		s3 += x3[i]
	}
	return d0, s0, d1, s1, d2, s2, d3, s3
}

func dotU4F32HighAndSumx4Scalar(q []byte, x []float32, stride int) (float32, float32, float32, float32, float32, float32, float32, float32) {
	var d0, s0, d1, s1, d2, s2, d3, s3 float32
	x0 := x[:len(q)]
	x1 := x[stride : stride+len(q)]
	x2 := x[2*stride : 2*stride+len(q)]
	x3 := x[3*stride : 3*stride+len(q)]
	for i, b := range q {
		qv := float32(b >> 4)
		d0 += qv * x0[i]
		s0 += x0[i]
		d1 += qv * x1[i]
		s1 += x1[i]
		d2 += qv * x2[i]
		s2 += x2[i]
		d3 += qv * x3[i]
		s3 += x3[i]
	}
	return d0, s0, d1, s1, d2, s2, d3, s3
}
