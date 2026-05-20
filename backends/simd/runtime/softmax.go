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
