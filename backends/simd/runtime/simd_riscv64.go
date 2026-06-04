//go:build riscv64

package simd

import "unsafe"

// riscv64 exposes SGEMM capability when RVV is available. SgemmNT uses the
// RVV dot-product kernel for each output cell; SgemmNN remains scalar-safe
// because B columns are strided in row-major layout.
const hasSgemmAsm = true

//go:noescape
func sgemmNT1x4Asm(a, b *float32, ldb int, k int, sums *float32)

//go:noescape
func sdotM4Asm(x, y []float32) float32

//go:noescape
func sdotM8Asm(x, y []float32) float32

//go:noescape
func sdotM4x2Asm(x, y []float32) float32

// SgemmNT computes C += alpha * A * B^T. Rows of A and B are contiguous, so
// each output cell uses the RVV dot-product kernel directly.
//
// NOTE: a register-blocked 1x4 RVV microkernel (sgemmNT1x4Asm) was implemented
// and validated, but benchmarks on the SpaceMIT K1 (in-order, single-issue RVV
// pipe) showed it is ~0.84x — slower than this per-cell path. F32 RVV on the K1
// is overhead/latency-bound around ~4.4 GFLOP/s and the simple kernel already
// reaches that ceiling, so the per-cell dispatch is retained. The real
// throughput lever is the int8 IME engine (backends/spacemit/ime2). The 1x4
// kernel is kept for the A/B benchmark in sgemm_bench_riscv64_test.go.
func SgemmNT(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	if !validRawSgemmArgs(m, n, k, a, b, c, lda, ldb, ldc, true) {
		return
	}
	for i := 0; i < m; i++ {
		aOff, okA := checkedFloat32ByteOffset(i * lda)
		if !okA {
			return
		}
		aRow := unsafe.Slice((*float32)(unsafe.Add(a, aOff)), k)
		for j := 0; j < n; j++ {
			bOff, okB := checkedFloat32ByteOffset(j * ldb)
			if !okB {
				return
			}
			bRow := unsafe.Slice((*float32)(unsafe.Add(b, bOff)), k)
			sum := sdotM4Asm(aRow, bRow)
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}

// sgemmNT1x4 is the register-blocked path, exposed for benchmarking/tests.
func sgemmNT1x4(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	if !validRawSgemmArgs(m, n, k, a, b, c, lda, ldb, ldc, true) {
		return
	}
	ldbBytes := ldb * 4
	var sums [4]float32
	for i := 0; i < m; i++ {
		aOff, okA := checkedFloat32ByteOffset(i * lda)
		if !okA {
			return
		}
		aPtr := (*float32)(unsafe.Add(a, aOff))
		aRow := unsafe.Slice(aPtr, k)
		j := 0
		for ; j+4 <= n; j += 4 {
			bOff, okB := checkedFloat32ByteOffset(j * ldb)
			if !okB {
				return
			}
			bPtr := (*float32)(unsafe.Add(b, bOff))
			sgemmNT1x4Asm(aPtr, bPtr, ldbBytes, k, &sums[0])
			storeF32(c, i*ldc+j+0, loadF32(c, i*ldc+j+0)+alpha*sums[0])
			storeF32(c, i*ldc+j+1, loadF32(c, i*ldc+j+1)+alpha*sums[1])
			storeF32(c, i*ldc+j+2, loadF32(c, i*ldc+j+2)+alpha*sums[2])
			storeF32(c, i*ldc+j+3, loadF32(c, i*ldc+j+3)+alpha*sums[3])
		}
		for ; j < n; j++ {
			bOff, okB := checkedFloat32ByteOffset(j * ldb)
			if !okB {
				return
			}
			bRow := unsafe.Slice((*float32)(unsafe.Add(b, bOff)), k)
			sum := sdotM4Asm(aRow, bRow)
			storeF32(c, i*ldc+j, loadF32(c, i*ldc+j)+alpha*sum)
		}
	}
}

// SgemmNN computes C += alpha * A * B. B columns are strided in row-major
// layout, so each column is packed into contiguous scratch once, then reused
// across all A rows with the RVV dot-product kernel.
func SgemmNN(m, n, k int, alpha float32, a, b, c unsafe.Pointer, lda, ldb, ldc int) {
	if !validRawSgemmArgs(m, n, k, a, b, c, lda, ldb, ldc, false) {
		return
	}
	bCol := make([]float32, k)
	for j := 0; j < n; j++ {
		for p := 0; p < k; p++ {
			bCol[p] = loadF32(b, p*ldb+j)
		}
		for i := 0; i < m; i++ {
			aOff, okA := checkedFloat32ByteOffset(i * lda)
			if !okA {
				return
			}
			aRow := unsafe.Slice((*float32)(unsafe.Add(a, aOff)), k)
			sum := sdotM4Asm(aRow, bCol)
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
