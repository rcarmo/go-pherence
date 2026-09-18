package simd

import "unsafe"

// SgemmNNPackedOverwriteTo computes C = A*B using caller-owned packing scratch.
// Unlike SgemmNNTo it overwrites C and has no alpha parameter. Scratch must hold
// k*16 floats and must not overlap A, B or C; C must not overlap either input.
// Invalid shapes return false without mutation. Full amd64 tiles use the
// packed microkernel; other platforms and small shapes use the NN kernel.
func SgemmNNPackedOverwriteTo(c, a, b, scratch []float32, m, n, k, lda, ldb, ldc int) bool {
	if !validSgemmSliceArgs(c, a, b, m, n, k, lda, ldb, ldc, false) || k > len(scratch)/16 {
		return false
	}
	for i := 0; i < m; i++ {
		clear(c[i*ldc : i*ldc+n])
	}
	if !hasPackedNN || !HasSgemmAsm || m < gebpMR || n < 16 {
		return SgemmNNTo(c, a, b, m, n, k, 1, lda, ldb, ldc)
	}
	fullRows := m / gebpMR * gebpMR
	fullCols := n / 16 * 16
	for col := 0; col < fullCols; col += 16 {
		for p := 0; p < k; p++ {
			copy(scratch[p*16:(p+1)*16], b[p*ldb+col:p*ldb+col+16])
		}
		for row := 0; row < fullRows; row += gebpMR {
			gebpMicroKernel(k, 1, unsafe.Pointer(&a[row*lda]), lda, unsafe.Pointer(&scratch[0]), unsafe.Pointer(&c[row*ldc+col]), ldc)
		}
	}
	if fullRows < m {
		SgemmNNTo(c[fullRows*ldc:], a[fullRows*lda:], b, m-fullRows, n, k, 1, lda, ldb, ldc)
	}
	if fullCols < n {
		SgemmNNTo(c[fullCols:], a, b[fullCols:], fullRows, n-fullCols, k, 1, lda, ldb, ldc)
	}
	return true
}
