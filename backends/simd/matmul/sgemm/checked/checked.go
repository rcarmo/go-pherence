package checked

import "unsafe"

// SgemmNTTo computes C += alpha * A * B^T for row-major A[m,lda],
// B[n,ldb], and C[m,ldc]. It validates slice capacities before invoking
// unsafe assembly paths and falls back to the scalar reference otherwise.
func SgemmNTTo(c, a, b []float32, m, n, k int, alpha float32, lda, ldb, ldc int) bool {
	if !validSgemmSliceArgs(c, a, b, m, n, k, lda, ldb, ldc, true) {
		return false
	}
	if HasSgemmAsm {
		SgemmNT(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), lda, ldb, ldc)
		return true
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += a[i*lda+p] * b[j*ldb+p]
			}
			c[i*ldc+j] += alpha * sum
		}
	}
	return true
}

// SgemmNNTo computes C += alpha * A * B for row-major A[m,lda], B[k,ldb],
// and C[m,ldc]. It validates slice capacities before unsafe assembly work.
func SgemmNNTo(c, a, b []float32, m, n, k int, alpha float32, lda, ldb, ldc int) bool {
	if !validSgemmSliceArgs(c, a, b, m, n, k, lda, ldb, ldc, false) {
		return false
	}
	if HasSgemmAsm {
		SgemmNN(m, n, k, alpha, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), lda, ldb, ldc)
		return true
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			sum := float32(0)
			for p := 0; p < k; p++ {
				sum += a[i*lda+p] * b[p*ldb+j]
			}
			c[i*ldc+j] += alpha * sum
		}
	}
	return true
}

func validSgemmSliceArgs(c, a, b []float32, m, n, k, lda, ldb, ldc int, nt bool) bool {
	if m <= 0 || n <= 0 || k <= 0 || lda < k || ldc < n {
		return false
	}
	if nt {
		if ldb < k {
			return false
		}
	} else if ldb < n {
		return false
	}
	aBase, okABase := checkedMulInt(m-1, lda)
	aNeed, okA := checkedAddInt(aBase, k)
	var bNeed int
	var okB bool
	if nt {
		bBase, okBBase := checkedMulInt(n-1, ldb)
		bNeed, okB = checkedAddInt(bBase, k)
		okB = okB && okBBase
	} else {
		bBase, okBBase := checkedMulInt(k-1, ldb)
		bNeed, okB = checkedAddInt(bBase, n)
		okB = okB && okBBase
	}
	cBase, okCBase := checkedMulInt(m-1, ldc)
	cNeed, okC := checkedAddInt(cBase, n)
	return okABase && okA && okB && okCBase && okC && len(a) >= aNeed && len(b) >= bNeed && len(c) >= cNeed
}
