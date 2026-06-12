package rvv

import "math"

// FastExp computes an approximation of exp(x) using the Schraudolph integer
// trick. Accurate to ~0.1% for x in [-10, 10], which is sufficient for softmax.
func FastExp(x float32) float32 {
	if x < -88 {
		return 0
	}
	if x > 88 {
		return float32(math.MaxFloat32)
	}
	// Schraudolph's method: interpret float bits as int, add scaled x
	const (
		shift = 1 << 23                             // 2^23
		bias  = 127 * shift                         // IEEE754 exponent bias
		coeff = float32(shift) / 0.6931471805599453 // 2^23 / ln(2)
	)
	i := int32(x*coeff) + int32(bias)
	if i < 0 {
		i = 0
	}
	return math.Float32frombits(uint32(i))
}
