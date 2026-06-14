package simd

import (
	"runtime"
	"sync"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// GemmRowsParallel computes out[batch, rows] = x[batch, cols] @ W[rows, cols]^T
// using goroutines across the output-row dimension. Numerics are identical to
// GemmRows (same per-element Sdot); only the work distribution differs.
func GemmRowsParallel(out, x, w []float32, batch, rows, cols int) bool {
	weightLen, okW := checked.MulInt(rows, cols)
	xLen, okX := checked.MulInt(batch, cols)
	outLen, okO := checked.MulInt(batch, rows)
	if batch <= 0 || rows <= 0 || cols <= 0 || !okW || !okX || !okO ||
		len(out) < outLen || len(x) < xLen || len(w) < weightLen {
		return false
	}
	nWorkers := runtime.GOMAXPROCS(0)
	if rows < 256 || nWorkers <= 1 {
		return GemmRows(out, x, w, batch, rows, cols)
	}
	if nWorkers > rows/64 {
		nWorkers = rows / 64
	}
	if nWorkers < 1 {
		nWorkers = 1
	}
	chunk := (rows + nWorkers - 1) / nWorkers
	var wg sync.WaitGroup
	for i := 0; i < nWorkers; i++ {
		start := i * chunk
		end := start + chunk
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(rs, re int) {
			defer wg.Done()
			for row := rs; row < re; row++ {
				wRow := w[row*cols : (row+1)*cols]
				for b := 0; b < batch; b++ {
					out[b*rows+row] = Sdot(x[b*cols:(b+1)*cols], wRow)
				}
			}
		}(start, end)
	}
	wg.Wait()
	return true
}

// SgemmNNParallelTo computes C[m,n] += alpha * A[m,k] @ B[k,n] using goroutines
// across the output-column dimension. Each worker runs the serial SgemmNNTo on
// a contiguous column block, so per-element accumulation order — and therefore
// the floating-point result — is identical to a single SgemmNNTo call.
func SgemmNNParallelTo(c, a, b []float32, m, n, k int, alpha float32, lda, ldb, ldc int) bool {
	if !validSgemmSliceArgs(c, a, b, m, n, k, lda, ldb, ldc, false) {
		return false
	}
	nWorkers := runtime.GOMAXPROCS(0)
	if n < 256 || nWorkers <= 1 {
		return SgemmNNTo(c, a, b, m, n, k, alpha, lda, ldb, ldc)
	}
	if nWorkers > n/64 {
		nWorkers = n / 64
	}
	if nWorkers < 1 {
		nWorkers = 1
	}
	chunk := (n + nWorkers - 1) / nWorkers
	// Align column chunks to 8 lanes so SIMD column kernels stay well-formed.
	chunk = (chunk + 7) &^ 7
	var wg sync.WaitGroup
	ok := true
	var mu sync.Mutex
	for col0 := 0; col0 < n; col0 += chunk {
		col1 := col0 + chunk
		if col1 > n {
			col1 = n
		}
		wg.Add(1)
		go func(c0, c1 int) {
			defer wg.Done()
			if !SgemmNNTo(c[c0:], a, b[c0:], m, c1-c0, k, alpha, lda, ldb, ldc) {
				mu.Lock()
				ok = false
				mu.Unlock()
			}
		}(col0, col1)
	}
	wg.Wait()
	return ok
}
