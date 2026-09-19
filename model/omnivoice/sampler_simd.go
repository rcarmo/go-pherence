package omnivoice

import (
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"math"
)

// logSoftmaxSIMDInPlace uses caller-owned disjoint scratch. Float64 summation
// stays sequential; exponentials and shifted inputs are rounded to float32.
// Exceptional rows use the original scalar policy. This is bounded numerical
// parity, not bitwise equivalence to the scalar float64 exponential path.
func logSoftmaxSIMDInPlace(x, scratch []float32) {
	if len(x) == 0 {
		return
	}
	max := x[0]
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 1) {
			logSoftmaxInPlace(x)
			return
		}
		if v > max {
			max = v
		}
	}
	if math.IsInf(float64(max), -1) {
		logSoftmaxInPlace(x)
		return
	}
	tmp := scratch[:len(x)]
	for i, v := range x {
		tmp[i] = v - max
	}
	simd.ExpF32To(tmp, tmp)
	var sum float64
	for _, v := range tmp {
		sum += float64(v)
	}
	logZ := float32(float64(max) + math.Log(sum))
	for i, v := range x {
		x[i] = v - logZ
	}
}
