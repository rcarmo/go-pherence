package simd

import "math"

// AffineF32InPlaceChecked replaces x with fused x*scale+shift, round-to-nearest
// even. The operation is exactly in-place: no partially overlapping source and
// destination slices exist. Empty x succeeds. NaN/Inf inputs or coefficients
// fail before writes; finite inputs may produce IEEE overflow to infinity.
// Uses AVX2/FMA on supported amd64; other CPUs use FMA32Scalar. The caller must
// retain Go's default floating-point environment; the amd64 assembly path
// rejects non-nearest rounding or DAZ/FTZ without changing input or FP control.
// No allocation or retained state. A single call is synchronous; callers that
// need cancellation should submit bounded blocks (SincNet uses <=4096 values).
func AffineF32InPlaceChecked(x []float32, scale, shift float32) bool {
	if !affineFinite(scale) || !affineFinite(shift) {
		return false
	}
	for _, v := range x {
		if !affineFinite(v) {
			return false
		}
	}
	if len(x) == 0 {
		return true
	}
	return affineF32InPlace(x, scale, shift)
}
func affineFinite(v float32) bool { return math.Float32bits(v)&0x7f800000 != 0x7f800000 }
func affineF32Scalar(x []float32, scale, shift float32) {
	for i, v := range x {
		x[i] = FMA32Scalar(v, scale, shift)
	}
}

// FMA32Scalar evaluates a*b+c with one float32 rounding for finite inputs.
// The exact float32 product fits float64. TwoSum recovers the addition residual;
// only a float32 midpoint can need correction before narrowing. This avoids
// the double-rounding error of float32(math.FMA(float64(a),...)). It also handles
// signed zero, subnormal and overflow boundaries in Go's default FP environment.
// NaN/Inf follow math.FMA classification; NaN payload identity is unspecified.
func FMA32Scalar(a, b, c float32) float32 {
	product := float64(float64(a) * float64(b))
	sum := float64(product + float64(c))
	if math.IsNaN(sum) || math.IsInf(sum, 0) {
		return float32(math.FMA(float64(a), float64(b), float64(c)))
	}
	part := float64(sum - product)
	residual := float64((product - float64(sum-part)) + (float64(c) - part))
	rounded := float32(sum)
	if residual == 0 || float64(rounded) == sum {
		return rounded
	}
	var midpoint float64
	if math.IsInf(float64(rounded), 0) {
		midpoint = math.Copysign(float64(math.MaxFloat32)+math.Ldexp(1, 103), sum)
	} else {
		direction := float32(math.Inf(1))
		if sum < float64(rounded) {
			direction = float32(math.Inf(-1))
		}
		neighbor := math.Nextafter32(rounded, direction)
		if math.IsInf(float64(neighbor), 0) {
			midpoint = math.Copysign(float64(math.MaxFloat32)+math.Ldexp(1, 103), sum)
		} else {
			midpoint = (float64(rounded) + float64(neighbor)) * .5
		}
	}
	if sum == midpoint {
		sum = math.Nextafter(sum, math.Copysign(math.Inf(1), residual))
	}
	return float32(sum)
}
