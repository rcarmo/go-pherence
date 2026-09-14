package simd

// FMAMatrixF32Checked overwrites dst[M,N] with A[M,K]*B[K,N], using ascending-K
// float32 FMA, positive-zero initial sums and no bias/alpha/beta. Unlike the
// general SGEMM API it checks exact extents, finiteness and output non-overlap.
// Bounds M1..256,N1..256,K1..4096 bound synchronous work. Read-only inputs may
// overlap. No allocation/retained state. Finite overflow may produce nonfinite
// output; callers decide whether to reject it. Requires round-to-nearest-even
// without DAZ/FTZ; amd64 checks those MXCSR fields before any writes. Exception
// masks are caller responsibility and are not checked/changed. Other paths use scalar
// FMA32Scalar. Feature admission is immutable, independent of HasSgemmAsm.
func FMAMatrixF32Checked(dst, a, b []float32, m, n, k int) bool {
	if m < 1 || m > 256 || n < 1 || n > 256 || k < 1 || k > 4096 || len(dst) != m*n || len(a) != m*k || len(b) != k*n || fmaColumnsOverlap(dst, a) || fmaColumnsOverlap(dst, b) {
		return false
	}
	for _, x := range [][]float32{a, b} {
		for _, v := range x {
			if !affineFinite(v) {
				return false
			}
		}
	}
	return fmaMatrixF32(dst, a, b, m, n, k)
}
func fmaMatrixScalar(dst, a, b []float32, m, n, k int) {
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var sum float32
			for p := 0; p < k; p++ {
				sum = FMA32Scalar(a[i*k+p], b[p*n+j], sum)
			}
			dst[i*n+j] = sum
		}
	}
}
