package simd

import (
	"math"
	"unsafe"
)

// expF32SimdAbsMaxBits gates the AVX2/FMA fast path to the finite range where
// the Cephes-style reduction plus polynomial approximation stays within the
// checked error target. Values outside this range fall back to scalar math.Exp.
const expF32SimdAbsMaxBits uint32 = 0x42000000 // 32.0f

// ExpF32To writes exp(src[i]) into dst and reports malformed inputs.
//
// Contract:
//   - len(dst) must equal len(src) and be non-zero
//   - dst may be exactly the same slice as src for in-place operation
//   - partially overlapping dst/src slices are rejected and return false
func ExpF32To(dst, src []float32) bool {
	if len(src) == 0 || len(dst) != len(src) || !expF32AliasOK(dst, src) {
		return false
	}
	expF32To(dst, src)
	return true
}

func expF32ToScalar(dst, src []float32) {
	for i, x := range src {
		dst[i] = expF32Scalar(x)
	}
}

func expF32Scalar(x float32) float32 {
	return float32(math.Exp(float64(x)))
}

func expF32FastPathBits(bits uint32) bool {
	return bits&0x7fffffff <= expF32SimdAbsMaxBits
}

func expF32AliasOK(dst, src []float32) bool {
	dp := uintptr(unsafe.Pointer(unsafe.SliceData(dst)))
	sp := uintptr(unsafe.Pointer(unsafe.SliceData(src)))
	if dp == sp {
		return true
	}
	dsz, okD := checkedFloat32ByteOffset(len(dst))
	ssz, okS := checkedFloat32ByteOffset(len(src))
	if !okD || !okS {
		return false
	}
	dEnd := dp + dsz
	sEnd := sp + ssz
	return dEnd <= sp || sEnd <= dp
}
