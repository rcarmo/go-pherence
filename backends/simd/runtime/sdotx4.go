package simd

// Sdotx4 computes four dot products between one weight row w and four
// contiguous input rows in x separated by stride. It is intended for batched
// GEMV-style inner loops that reuse the same dequantized row across several
// positions.
func Sdotx4(w, x []float32, stride int) (float32, float32, float32, float32, bool) {
	if len(w) == 0 || stride < len(w) || len(x) < 3*stride+len(w) {
		return 0, 0, 0, 0, false
	}
	d0, d1, d2, d3 := sdotx4(w, x, stride)
	return d0, d1, d2, d3, true
}

func sdotx4Scalar(w, x []float32, stride int) (float32, float32, float32, float32) {
	var d0, d1, d2, d3 float32
	x0 := x[:len(w)]
	x1 := x[stride : stride+len(w)]
	x2 := x[2*stride : 2*stride+len(w)]
	x3 := x[3*stride : 3*stride+len(w)]
	for i, wv := range w {
		d0 += wv * x0[i]
		d1 += wv * x1[i]
		d2 += wv * x2[i]
		d3 += wv * x3[i]
	}
	return d0, d1, d2, d3
}
