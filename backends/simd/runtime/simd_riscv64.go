//go:build riscv64

package simd

import "unsafe"

// Keep SGEMM assembly capability disabled until real RVV kernels are wired in.
// The raw entrypoints below are scalar-safe so direct architecture builds do
// useful work instead of the generic non-SIMD no-op path.
const hasSgemmAsm = false

func Saxpy(alpha float32, x []float32, y []float32) { saxpyScalar(alpha, x, y) }

// SgemmNT computes C += alpha * A * B^T using a scalar riscv64 fallback.
func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	if !validRawSgemmArgs(m, n, k, a, b, c, lda, ldb, ldc, true) {
		return
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += loadF32(a, i*lda+p) * loadF32(b, j*ldb+p)
			}
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}

// SgemmNN computes C += alpha * A * B using a scalar riscv64 fallback.
func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	if !validRawSgemmArgs(m, n, k, a, b, c, lda, ldb, ldc, false) {
		return
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += loadF32(a, i*lda+p) * loadF32(b, p*ldb+j)
			}
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}

func validRawSgemmArgs(m, n, k int, a, b, c unsafe.Pointer, lda, ldb, ldc int, nt bool) bool {
	if m <= 0 || n <= 0 || k <= 0 || a == nil || b == nil || c == nil || lda < k || ldc < n {
		return false
	}
	if nt {
		return ldb >= k
	}
	return ldb >= n
}

func loadF32(base unsafe.Pointer, idx int) float32 {
	off, ok := checkedFloat32ByteOffset(idx)
	if !ok {
		return 0
	}
	return *(*float32)(unsafe.Add(base, off))
}

func storeF32(base unsafe.Pointer, idx int, v float32) {
	off, ok := checkedFloat32ByteOffset(idx)
	if !ok {
		return
	}
	*(*float32)(unsafe.Add(base, off)) = v
}
