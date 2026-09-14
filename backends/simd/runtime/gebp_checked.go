package simd

import "unsafe"

// SgemmNTPackedTo computes C += alpha*A*B^T with caller-owned k*16 scratch.
// Unlike the legacy pointer API, all slice footprints are checked and partial
// row tiles never read beyond A. The full 6x16 amd64 / 4x16 arm64 microkernel
// shares each packed B panel across rows. Unsupported platforms use checked
// scalar/standard dispatch without touching scratch. No allocation on success.
func SgemmNTPackedTo(c, a, b, scratch []float32, m, n, k int, alpha float32, lda, ldb, ldc int) bool {
	if !validSgemmSliceArgs(c, a, b, m, n, k, lda, ldb, ldc, true) {
		return false
	}
	if k > int(^uint(0)>>1)/gebpNR || len(scratch) < k*gebpNR {
		return false
	}
	if !HasSgemmAsm || m < gebpMR {
		return SgemmNTTo(c, a, b, m, n, k, alpha, lda, ldb, ldc)
	}
	bp := scratch[:k*gebpNR]
	for jj := 0; jj < n; jj += gebpNR {
		nr := min(gebpNR, n-jj)
		if nr < gebpNR {
			// Tail columns use checked dot-product GEMM; no fake full-size B tile.
			if !SgemmNTTo(c[jj:], a, b[jj*ldb:], m, nr, k, alpha, lda, ldb, ldc) {
				return false
			}
			continue
		}
		packBNT(b, ldb, jj, nr, k, bp)
		ii := 0
		for ; ii+gebpMR <= m; ii += gebpMR {
			gebpMicroKernel(k, alpha, unsafe.Pointer(&a[ii*lda]), lda, unsafe.Pointer(&bp[0]), unsafe.Pointer(&c[ii*ldc+jj]), ldc)
		}
		if ii < m {
			if !SgemmNTTo(c[ii*ldc+jj:], a[ii*lda:], b[jj*ldb:], m-ii, nr, k, alpha, lda, ldb, ldc) {
				return false
			}
		}
	}
	return true
}
