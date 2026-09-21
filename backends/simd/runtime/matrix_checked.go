package simd

import "github.com/rcarmo/go-pherence/internal/checked"

const matMulTransposeBlock = 256

// MatMul overwrites dst[m,n] with op(A)[m,k] * op(B)[k,n], where A is stored as
// [m,k] when transposeA is false and [k,m] when transposeA is true, and B is
// stored as [k,n] when transposeB is false and [n,k] when transposeB is true.
//
// Invalid dimensions, overflowing footprints, short slices, or any overlap
// between dst and a/b return false without writing dst.
func MatMul(dst, a, b []float32, m, n, k int, transposeA, transposeB bool) bool {
	needDst, needA, needB, ok := validMatMulArgs(dst, a, b, m, n, k)
	if !ok {
		return false
	}
	dst = dst[:needDst]
	a = a[:needA]
	b = b[:needB]
	if !float32SlicesDisjoint(dst, a) || !float32SlicesDisjoint(dst, b) {
		return false
	}

	clear(dst)
	if !transposeA {
		if transposeB {
			return SgemmNTTo(dst, a, b, m, n, k, 1, k, k, n)
		}
		return SgemmNNTo(dst, a, b, m, n, k, 1, k, n, n)
	}
	if transposeB {
		matMulTransposeAT(dst, a, b, m, n, k)
		return true
	}
	matMulTransposeAN(dst, a, b, m, n, k)
	return true
}

func validMatMulArgs(dst, a, b []float32, m, n, k int) (needDst, needA, needB int, ok bool) {
	if m <= 0 || n <= 0 || k <= 0 {
		return 0, 0, 0, false
	}
	needDst, okDst := checked.MulInt(m, n)
	needA, okA := checked.MulInt(m, k)
	needB, okB := checked.MulInt(k, n)
	if !okDst || !okA || !okB {
		return 0, 0, 0, false
	}
	if len(dst) < needDst || len(a) < needA || len(b) < needB {
		return 0, 0, 0, false
	}
	return needDst, needA, needB, true
}

func matMulTransposeAN(dst, a, b []float32, m, n, k int) {
	for p := 0; p < k; p++ {
		aCol := a[p*m : (p+1)*m]
		bRow := b[p*n : (p+1)*n]
		for i, alpha := range aCol {
			if alpha == 0 {
				continue
			}
			Saxpy(alpha, bRow, dst[i*n:(i+1)*n])
		}
	}
}

func matMulTransposeAT(dst, a, b []float32, m, n, k int) {
	var scratch [matMulTransposeBlock]float32
	for i := 0; i < m; i++ {
		dstRow := dst[i*n : (i+1)*n]
		for p0 := 0; p0 < k; p0 += matMulTransposeBlock {
			kb := k - p0
			if kb > matMulTransposeBlock {
				kb = matMulTransposeBlock
			}
			for p := 0; p < kb; p++ {
				scratch[p] = a[(p0+p)*m+i]
			}
			x := scratch[:kb]
			for j := 0; j < n; j++ {
				dstRow[j] += Sdot(x, b[j*k+p0:j*k+p0+kb])
			}
		}
	}
}
