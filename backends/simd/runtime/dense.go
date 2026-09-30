package simd

import (
	"runtime"
	"unsafe"
)

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
// The blocked kernel currently requires contiguous rows and output. The
// 576-row cap covers bounded Nemotron diarisation windows (maximum 541 rows)
// while retaining the checked serial path for larger/strided operations.
func DenseNTTo(c, a, b []float32, m, n, k int, alpha float32, lda, ldb, ldc int) bool {
	if !validSgemmSliceArgs(c, a, b, m, n, k, lda, ldb, ldc, true) {
		return false
	}
	if HasSgemmAsm && m > 1 && m <= 576 && n >= 64 && k >= 64 && lda == k && ldb == k && ldc == n {
		// Tiny streaming query blocks still have large output-channel work.
		// Partition the existing blocked kernel, preserving K accumulation
		// order and accumulating destination semantics. Larger-M callers keep
		// their existing dispatch to avoid nested head/batch parallelism.
		if m <= 5 && n >= 1024 && k >= 1024 && alpha == 1 && runtime.GOMAXPROCS(0) > 1 {
			return sgemmNTBlockedParallelTo(c, a, b, m, n, k, min(runtime.GOMAXPROCS(0), 4))
		}
		SgemmNTBlockedFMA(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), lda, ldb, ldc)
		return true
	}
	return SgemmNTTo(c, a, b, m, n, k, alpha, lda, ldb, ldc)
}
