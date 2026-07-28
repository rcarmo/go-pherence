package simd

import "unsafe"

// DenseNNTo applies shape-aware checked dispatch for C += alpha*A*B.
func DenseNNTo(c, a, b []float32, m, n, k int, alpha float32, lda, ldb, ldc int) bool {
	if !validSgemmSliceArgs(c, a, b, m, n, k, lda, ldb, ldc, false) {
		return false
	}
	if m > 1 && n >= 256 && int64(m)*int64(n)*int64(k) >= 1<<22 {
		return SgemmNNParallelTo(c, a, b, m, n, k, alpha, lda, ldb, ldc)
	}
	return SgemmNNTo(c, a, b, m, n, k, alpha, lda, ldb, ldc)
}

// DenseNTTo applies shape-aware checked dispatch for C += alpha*A*B^T.
// The blocked kernel currently requires contiguous rows and output.
func DenseNTTo(c, a, b []float32, m, n, k int, alpha float32, lda, ldb, ldc int) bool {
	if !validSgemmSliceArgs(c, a, b, m, n, k, lda, ldb, ldc, true) {
		return false
	}
	if HasSgemmAsm && m > 1 && m <= 256 && n >= 64 && k >= 64 && lda == k && ldb == k && ldc == n {
		SgemmNTBlockedFMA(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), lda, ldb, ldc)
		return true
	}
	return SgemmNTTo(c, a, b, m, n, k, alpha, lda, ldb, ldc)
}
