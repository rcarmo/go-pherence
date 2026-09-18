package simd

import (
	"math"
)

const (
	// Conservative fast-path window:
	//   - |x| <= 8 keeps the internal exp(-0.5*x*x) argument inside [-32, 0],
	//     matching the checked exp polynomial range used by the AVX2/FMA kernel.
	//   - |x| >= 2^-12 leaves signed zero and tiny-tail behavior to scalar math.Erf.
	geluErfF32SimdAbsMaxBits uint32 = 0x41000000 // 8.0f
	geluErfF32SimdAbsMinBits uint32 = 0x39800000 // 2^-12
)

// GELUErfF32To writes exact-erf GELU(src[i]) into dst and reports malformed inputs.
//
// Formula:
//
//	dst[i] = 0.5 * x * (1 + erf(x / sqrt(2)))
//
// The amd64 AVX2/FMA kernel uses the Abramowitz-Stegun 7.1.26 erf approximation
// (published max |erf error| about 1.5e-7) composed with a bounded Cephes-style
// float32 exp approximation for exp(-0.5*x*x). Inputs outside the validated SIMD
// window, plus signed-zero/tiny-tail cases, fall back to the scalar formula.
//
// Contract:
//   - len(dst) must equal len(src) and be non-zero
//   - dst may be exactly the same slice as src for in-place operation
//   - partially overlapping dst/src slices are rejected and return false
func GELUErfF32To(dst, src []float32) bool {
	if len(src) == 0 || len(dst) != len(src) || !expF32AliasOK(dst, src) {
		return false
	}
	geluErfF32To(dst, src)
	return true
}

func geluErfF32ToScalar(dst, src []float32) {
	for i, x := range src {
		dst[i] = geluErfF32Scalar(x)
	}
}

func geluErfF32Scalar(x float32) float32 {
	const invSqrt2 = 0.7071067811865476
	return 0.5 * x * (1 + float32(math.Erf(float64(x)*invSqrt2)))
}

func geluErfF32FastPathBits(bits uint32) bool {
	abs := bits & 0x7fffffff
	return abs >= geluErfF32SimdAbsMinBits && abs <= geluErfF32SimdAbsMaxBits
}
