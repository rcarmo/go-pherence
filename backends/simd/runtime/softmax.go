package simd

import "math"

// SoftmaxInPlace normalizes x in-place with a numerically stable row softmax.
func SoftmaxInPlace(x []float32) bool {
	if len(x) == 0 {
		return false
	}
	mx := x[0]
	for _, v := range x[1:] {
		if v > mx {
			mx = v
		}
	}
	sum := float32(0)
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - mx)))
		sum += x[i]
	}
	if sum == 0 {
		return false
	}
	inv := 1 / sum
	for i := range x {
		x[i] *= inv
	}
	return true
}

// SoftmaxRowsInPlace normalizes a row-major [rows, cols] matrix in-place.
func SoftmaxRowsInPlace(x []float32, rows, cols int) bool {
	total, ok := checkedMulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(x) < total {
		return false
	}
	for r := 0; r < rows; r++ {
		if !SoftmaxInPlace(x[r*cols : (r+1)*cols]) {
			return false
		}
	}
	return true
}
