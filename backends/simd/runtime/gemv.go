package simd

import (
	"runtime"
	"sync"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// GemvRows computes out[rows] = W[rows,cols] · x[cols], where W is row-major.
// It is the backend-owned dense F32 GEMV reference used by CPU fallbacks and
// future optimized dispatch paths.
func GemvRows(out, x, w []float32, rows, cols int) bool {
	weightLen, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(out) < rows || len(x) < cols || len(w) < weightLen {
		return false
	}
	x = x[:cols]
	for row := 0; row < rows; row++ {
		out[row] = Sdot(x, w[row*cols:(row+1)*cols])
	}
	return true
}

// GemvRowsBF16 computes out[rows] = W_bf16[rows,cols] · x[cols], where W is
// row-major BF16 ([]uint16) and x/out are F32. Uses BF16DotF32 with F32 input.
// On amd64 with AVX2, this avoids full BF16→F32 weight decode.
func GemvRowsBF16(out []float32, x []float32, w []uint16, rows, cols int) bool {
	weightLen, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(out) < rows || len(x) < cols || len(w) < weightLen {
		return false
	}
	x = x[:cols]
	row := 0
	for ; row+4 <= rows; row += 4 {
		d0, d1, d2, d3, ok := BF16DotF32x4(w[row*cols:], x, cols)
		if !ok {
			return false
		}
		out[row+0] = d0
		out[row+1] = d1
		out[row+2] = d2
		out[row+3] = d3
	}
	for ; row < rows; row++ {
		out[row] = BF16DotF32(w[row*cols:(row+1)*cols], x)
	}
	return true
}

// GemvRowsBF16BF16 computes out[rows] = W_bf16[rows,cols] · x_bf16[cols],
// where both W and x are BF16. Uses BF16DotAsm when available (AVX2).
// Output is F32. Fastest path for non-resident weight layers.
func GemvRowsBF16BF16(out []float32, x []uint16, w []uint16, rows, cols int) bool {
	weightLen, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(out) < rows || len(x) < cols || len(w) < weightLen {
		return false
	}
	x = x[:cols]
	row := 0
	for ; row+4 <= rows; row += 4 {
		d0, d1, d2, d3, ok := BF16DotBF16x4(w[row*cols:], x, cols)
		if !ok {
			return false
		}
		out[row+0] = d0
		out[row+1] = d1
		out[row+2] = d2
		out[row+3] = d3
	}
	for ; row < rows; row++ {
		out[row] = BF16DotAsm(w[row*cols:(row+1)*cols], x)
	}
	return true
}

// GemvCols computes out[cols] = x[rows] · W[rows,cols], where W is row-major.
func GemvCols(out, x, w []float32, rows, cols int) bool {
	weightLen, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(out) < cols || len(x) < rows || len(w) < weightLen {
		return false
	}
	for col := 0; col < cols; col++ {
		sum := float32(0)
		for row := 0; row < rows; row++ {
			sum += x[row] * w[row*cols+col]
		}
		out[col] = sum
	}
	return true
}

// GemmRows computes out[batch,rows] = x[batch,cols] @ W[rows,cols]^T.
func GemmRows(out, x, w []float32, batch, rows, cols int) bool {
	xLen, okX := checked.MulInt(batch, cols)
	outLen, okOut := checked.MulInt(batch, rows)
	weightLen, okW := checked.MulInt(rows, cols)
	if batch <= 0 || rows <= 0 || cols <= 0 || !okX || !okOut || !okW || len(out) < outLen || len(x) < xLen || len(w) < weightLen {
		return false
	}
	for b := 0; b < batch; b++ {
		if !GemvRows(out[b*rows:(b+1)*rows], x[b*cols:(b+1)*cols], w[:weightLen], rows, cols) {
			return false
		}
	}
	return true
}

// GemmCols computes out[batch,cols] = x[batch,rows] @ W[rows,cols].
func GemmCols(out, x, w []float32, batch, rows, cols int) bool {
	xLen, okX := checked.MulInt(batch, rows)
	outLen, okOut := checked.MulInt(batch, cols)
	weightLen, okW := checked.MulInt(rows, cols)
	if batch <= 0 || rows <= 0 || cols <= 0 || !okX || !okOut || !okW || len(out) < outLen || len(x) < xLen || len(w) < weightLen {
		return false
	}
	for b := 0; b < batch; b++ {
		if !GemvCols(out[b*cols:(b+1)*cols], x[b*rows:(b+1)*rows], w[:weightLen], rows, cols) {
			return false
		}
	}
	return true
}

// GemvRowsParallel computes out[rows] = W[rows,cols] · x[cols] using
// goroutines to parallelize across rows. Falls back to GemvRows for small M.
func GemvRowsParallel(out, x, w []float32, rows, cols int) bool {
	weightLen, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(out) < rows || len(x) < cols || len(w) < weightLen {
		return false
	}
	if rows < 256 {
		return GemvRows(out, x, w, rows, cols)
	}
	x = x[:cols]
	nWorkers := runtime.GOMAXPROCS(0)
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
		go func(s, e int) {
			defer wg.Done()
			for row := s; row < e; row++ {
				out[row] = Sdot(x, w[row*cols:(row+1)*cols])
			}
		}(start, end)
	}
	wg.Wait()
	return true
}

// GemvRowsBF16BF16Parallel computes out[rows] = W[rows,cols] · x[cols] using
// BF16 dot products across multiple goroutines.
func GemvRowsBF16BF16Parallel(out []float32, x []uint16, w []uint16, rows, cols int) bool {
	weightLen, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(out) < rows || len(x) < cols || len(w) < weightLen {
		return false
	}
	x = x[:cols]
	nWorkers := runtime.GOMAXPROCS(0)
	if nWorkers > 6 {
		nWorkers = 6
	}
	if rows < nWorkers*64 {
		return GemvRowsBF16BF16(out, x, w, rows, cols)
	}
	chunk := (rows + nWorkers - 1) / nWorkers
	var wg sync.WaitGroup
	for wi := 0; wi < nWorkers; wi++ {
		start := wi * chunk
		end := start + chunk
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			row := s
			for ; row+4 <= e; row += 4 {
				d0, d1, d2, d3, ok := BF16DotBF16x4(w[row*cols:], x, cols)
				if !ok {
					return
				}
				out[row+0] = d0
				out[row+1] = d1
				out[row+2] = d2
				out[row+3] = d3
			}
			for ; row < e; row++ {
				out[row] = BF16DotAsm(w[row*cols:(row+1)*cols], x)
			}
		}(start, end)
	}
	wg.Wait()
	return true
}
