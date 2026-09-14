package simd

import "math"

// SoftmaxSIMDInPlace uses the bounded-error ExpF32To kernel. Summation remains
// float32 and sequential. Exceptional input semantics match SoftmaxInPlace.
// Unlike SoftmaxInPlace this is approximate on AVX2/FMA; not bitwise equivalent.
func SoftmaxSIMDInPlace(x []float32) bool {
	if len(x) == 0 {
		return false
	}
	m := x[0]
	for _, v := range x[1:] {
		if v > m {
			m = v
		}
	}
	for i := range x {
		x[i] -= m
	}
	if !ExpF32To(x, x) {
		return false
	}
	var sum float32
	for _, v := range x {
		sum += v
	}
	if sum == 0 || math.IsNaN(float64(sum)) || math.IsInf(float64(sum), 0) {
		return false
	}
	VecScale(x, x, 1/sum)
	return true
}
