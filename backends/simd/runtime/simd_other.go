//go:build !amd64 && !arm64 && !riscv64

package simd

import "unsafe"

const hasSgemmAsm = false

func Sdot(x, y []float32) float32 { return sdotScalar(x, y) }

func Saxpy(alpha float32, x []float32, y []float32) { saxpyScalar(alpha, x, y) }

// SgemmNT computes C += alpha * A * B^T using a scalar fallback.
func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	if !validRawSgemmArgsOther(m, n, k, a, b, c, lda, ldb, ldc, true) {
		return
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				aOff, okA := checkedFloat32ByteOffset(i*lda + p)
				bOff, okB := checkedFloat32ByteOffset(j*ldb + p)
				if !okA || !okB {
					return
				}
				sum += *(*float32)(unsafe.Add(a, aOff)) * *(*float32)(unsafe.Add(b, bOff))
			}
			cOff, okC := checkedFloat32ByteOffset(i*ldc + j)
			if !okC {
				return
			}
			cv := (*float32)(unsafe.Add(c, cOff))
			*cv += alpha * sum
		}
	}
}

// SgemmNN computes C += alpha * A * B using a scalar fallback.
func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	if !validRawSgemmArgsOther(m, n, k, a, b, c, lda, ldb, ldc, false) {
		return
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				aOff, okA := checkedFloat32ByteOffset(i*lda + p)
				bOff, okB := checkedFloat32ByteOffset(p*ldb + j)
				if !okA || !okB {
					return
				}
				sum += *(*float32)(unsafe.Add(a, aOff)) * *(*float32)(unsafe.Add(b, bOff))
			}
			cOff, okC := checkedFloat32ByteOffset(i*ldc + j)
			if !okC {
				return
			}
			cv := (*float32)(unsafe.Add(c, cOff))
			*cv += alpha * sum
		}
	}
}

func validRawSgemmArgsOther(m, n, k int, a, b, c unsafe.Pointer, lda, ldb, ldc int, nt bool) bool {
	if m <= 0 || n <= 0 || k <= 0 || a == nil || b == nil || c == nil || lda < k || ldc < n {
		return false
	}
	if nt {
		return ldb >= k
	}
	return ldb >= n
}
