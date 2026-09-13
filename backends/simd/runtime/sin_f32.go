package simd

import "math"

// sinF32SimdAbsMaxBits gates the AVX2/FMA fast path to the finite range where
// the range reduction x = k*(pi/2) + r with k = round(x*2/pi) is exact enough
// in float32, leaving r in [-pi/4, pi/4]. On that interval we use the
// truncated Maclaurin polynomials
//
//	sin(r) ≈ r * (1 + r²*(-1/6 + r²*(1/120 + r²*(-1/5040 + r²*(1/362880 - r²/39916800)))))
//	cos(r) ≈ 1 + r²*(-1/2 + r²*(1/24 + r²*(-1/720 + r²*(1/40320 - r²/3628800))))
//
// With FMA evaluation and the checked reduction constants below, exhaustive
// dense+random tests over finite float32 inputs in |x| <= 32 stay well below
// the required 2e-6 absolute-error target.
const sinF32SimdAbsMaxBits uint32 = 0x42000000 // 32.0f

// SinF32To writes sin(src[i]) into dst and reports malformed inputs.
//
// Contract:
//   - len(dst) must equal len(src) and be non-zero
//   - dst may be exactly the same slice as src for in-place operation
//   - partially overlapping dst/src slices are rejected and return false
func SinF32To(dst, src []float32) bool {
	if len(src) == 0 || len(dst) != len(src) || !expF32AliasOK(dst, src) {
		return false
	}
	sinF32To(dst, src)
	return true
}

func sinF32ToScalar(dst, src []float32) {
	for i, x := range src {
		dst[i] = sinF32Scalar(x)
	}
}

func sinF32Scalar(x float32) float32 {
	return float32(math.Sin(float64(x)))
}

func sinF32FastPathBits(bits uint32) bool {
	return bits != 0x80000000 && bits&0x7fffffff <= sinF32SimdAbsMaxBits
}
