package simd

import "unsafe"

// FMAColumnsF32Checked computes dst[j] = sum_k x[k*len(dst)+j]*weight[k]
// with one float32 FMA per k in ascending order, starting at positive zero.
// Unlike a dot-product reduction, vector lanes span independent output columns;
// the scalar fallback and SIMD path have identical reduction order. No bias.
// Empty output is accepted only with empty x; an empty reduction produces zeros.
// All extents, input finiteness and output non-overlap are checked before writes.
// Read-only x/weight may overlap. Finite overflow is allowed. No allocation or
// retained state. Requires the default FP environment (assembly checks MXCSR).
// Calls are synchronous: cancellation belongs at caller-selected tile boundaries.
func FMAColumnsF32Checked(dst, x, weight []float32) bool {
	if len(dst) == 0 {
		if len(x) != 0 {
			return false
		}
		for _, w := range weight {
			if !affineFinite(w) {
				return false
			}
		}
		return true
	}
	if len(x)%len(dst) != 0 || len(x)/len(dst) != len(weight) || fmaColumnsOverlap(dst, x) || fmaColumnsOverlap(dst, weight) {
		return false
	}
	for _, row := range [][]float32{x, weight} {
		for _, v := range row {
			if !affineFinite(v) {
				return false
			}
		}
	}
	return fmaColumnsF32(dst, x, weight)
}

func fmaColumnsOverlap(a, b []float32) bool {
	if len(a) == 0 || len(b) == 0 {
		return false
	}
	a0, b0 := uintptr(unsafe.Pointer(&a[0])), uintptr(unsafe.Pointer(&b[0]))
	return a0 < b0+uintptr(len(b))*4 && b0 < a0+uintptr(len(a))*4
}

func fmaColumnsScalar(dst, x, weight []float32) {
	for j := range dst {
		var sum float32
		for k, w := range weight {
			sum = FMA32Scalar(x[k*len(dst)+j], w, sum)
		}
		dst[j] = sum
	}
}
