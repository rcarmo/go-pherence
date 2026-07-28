package simd

import (
	"runtime"
	"sync"
	"unsafe"

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
	// Prompt/prefill matrices benefit from output tiling: each activation load
	// feeds multiple weight rows, as in llamafile's multi-output kernels. Keep
	// batch=1 on the lower-latency dot/GEMV path used by autoregressive decode.
	if HasSgemmAsm && batch > 1 && batch <= 256 && rows >= 64 && cols >= 64 {
		clear(out[:outLen])
		return sgemmNTBlockedParallelTo(out[:outLen], x[:xLen], w[:weightLen], batch, rows, cols, nWorkers)
	}
	return gemmRowsParallelDots(out, x, w, batch, rows, cols)
}

func gemmRowsParallelDots(out, x, w []float32, batch, rows, cols int) bool {
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

// GemmRowsBF16Parallel computes out[batch, rows] = x[batch, cols] @
// W_bf16[rows, cols]^T. It keeps checkpoint weights compressed and dispatches
// each dot product to the architecture-specific BF16 SIMD kernel.
func GemmRowsBF16Parallel(out, x []float32, w []uint16, batch, rows, cols int) bool {
	weightLen, okW := checked.MulInt(rows, cols)
	xLen, okX := checked.MulInt(batch, cols)
	outLen, okO := checked.MulInt(batch, rows)
	if batch <= 0 || rows <= 0 || cols <= 0 || !okW || !okX || !okO ||
		len(out) < outLen || len(x) < xLen || len(w) < weightLen {
		return false
	}
	nWorkers := runtime.GOMAXPROCS(0)
	if rows < 256 || nWorkers <= 1 {
		for b := 0; b < batch; b++ {
			if !GemvRowsBF16(out[b*rows:(b+1)*rows], x[b*cols:(b+1)*cols], w, rows, cols) {
				return false
			}
		}
		return true
	}
	if nWorkers > rows/64 {
		nWorkers = rows / 64
	}
	if nWorkers < 1 {
		nWorkers = 1
	}
	chunk := (rows + nWorkers - 1) / nWorkers
	var wg sync.WaitGroup
	for worker := 0; worker < nWorkers; worker++ {
		start := worker * chunk
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
			row := rs
			for ; row+4 <= re; row += 4 {
				wRows := w[row*cols:]
				for b := 0; b < batch; b++ {
					d0, d1, d2, d3, ok := BF16DotF32x4(wRows, x[b*cols:(b+1)*cols], cols)
					if !ok {
						return
					}
					base := b*rows + row
					out[base+0] = d0
					out[base+1] = d1
					out[base+2] = d2
					out[base+3] = d3
				}
			}
			for ; row < re; row++ {
				wRow := w[row*cols : (row+1)*cols]
				for b := 0; b < batch; b++ {
					out[b*rows+row] = BF16DotF32(wRow, x[b*cols:(b+1)*cols])
				}
			}
		}(start, end)
	}
	wg.Wait()
	return true
}

// sgemmNTBlockedParallelTo computes C=A*B^T by assigning contiguous output-row
// tiles to workers. Each worker reuses activation vectors across pairs of weight
// rows and K blocks; no worker shares output cache lines with another.
func sgemmNTBlockedParallelTo(c, a, b []float32, m, n, k, nWorkers int) bool {
	if !validSgemmSliceArgs(c, a, b, m, n, k, k, k, n, true) || !HasSgemmAsm {
		return false
	}
	if nWorkers < 1 {
		nWorkers = 1
	}
	const tileRows = 64
	tiles := (n + tileRows - 1) / tileRows
	if nWorkers > tiles {
		nWorkers = tiles
	}
	if nWorkers <= 1 {
		SgemmNTBlockedFMA(m, n, k, 1, unsafe.Pointer(&a[0]), unsafe.Pointer(&b[0]), unsafe.Pointer(&c[0]), k, k, n)
		return true
	}
	var wg sync.WaitGroup
	for worker := 0; worker < nWorkers; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for tile := worker; tile < tiles; tile += nWorkers {
				row0 := tile * tileRows
				rowN := tileRows
				if row0+rowN > n {
					rowN = n - row0
				}
				SgemmNTBlockedFMA(m, rowN, k, 1,
					unsafe.Pointer(&a[0]), unsafe.Pointer(&b[row0*k]), unsafe.Pointer(&c[row0]),
					k, k, n)
			}
		}(worker)
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
