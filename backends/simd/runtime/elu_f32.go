package simd

import (
	"math"
)

const (
	// ELU uses the AVX2/FMA expm1 kernel only for negative finite inputs in
	// [-16, -2^-20). Smaller magnitudes use scalar exp/subtract to preserve
	// its cancellation rounding. Nonnegative SIMD lanes pass through exactly.
	eluF32SimdMin        = float32(-16)
	eluF32TinyScalarMax  = float32(0x1p-20)
	eluF32NearZeroAbsTol = 1e-7
)

// ELUF32To writes alpha=1 ELU(src[i]) into dst and reports malformed inputs.
//
// Reference semantics:
//   - !(x < 0): dst[i] = x exactly, preserving signed zero, +Inf and NaN bits
//   - x < 0:  dst[i] = float32(math.Exp(float64(x)) - 1)
//
// Contract:
//   - len(dst) must equal len(src) and be non-zero
//   - dst may be exactly the same slice as src for in-place operation
//   - partially overlapping dst/src slices are rejected and return false
func ELUF32To(dst, src []float32) bool {
	if len(src) == 0 || len(dst) != len(src) || !expF32AliasOK(dst, src) {
		return false
	}
	eluF32To(dst, src)
	return true
}

func eluF32ToScalar(dst, src []float32) {
	for i, x := range src {
		dst[i] = eluF32Scalar(x)
	}
}

func eluF32Scalar(x float32) float32 {
	if !(x < 0) {
		return x
	}
	return float32(math.Exp(float64(x)) - 1)
}

func eluF32FastPath(x float32) bool {
	return x < -eluF32TinyScalarMax && x >= eluF32SimdMin
}
