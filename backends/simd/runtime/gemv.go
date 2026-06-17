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
	for row := 0; row < rows; row++ {
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
	for row := 0; row < rows; row++ {
		out[row] = BF16DotAsm(w[row*cols:(row+1)*cols], x)
	}
	return true
}

// GemvRowsBF16BF16GGMLAVX2Order computes BF16 rows using the same visible
// reduction order as ggml's AVX2 ggml_vec_dot_bf16: four 8-lane F32
// accumulators over 32-element chunks followed by the same horizontal sum shape.
func GemvRowsBF16BF16GGMLAVX2Order(out []float32, x []uint16, w []uint16, rows, cols int) bool {
	weightLen, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(out) < rows || len(x) < cols || len(w) < weightLen {
		return false
	}
	x = x[:cols]
	for row := 0; row < rows; row++ {
		out[row] = BF16DotGGMLAVX2Order(w[row*cols:(row+1)*cols], x)
	}
	return true
}

func BF16DotGGMLAVX2Order(x, y []uint16) float32 {
	n := len(x)
	if len(y) < n {
		n = len(y)
	}
	var c [4][8]float32
	i := 0
	for ; i+32 <= n; i += 32 {
		for lane := 0; lane < 8; lane++ {
			c[0][lane] += BF16ToF32(x[i+lane]) * BF16ToF32(y[i+lane])
			c[1][lane] += BF16ToF32(x[i+8+lane]) * BF16ToF32(y[i+8+lane])
			c[2][lane] += BF16ToF32(x[i+16+lane]) * BF16ToF32(y[i+16+lane])
			c[3][lane] += BF16ToF32(x[i+24+lane]) * BF16ToF32(y[i+24+lane])
		}
	}
	var v [8]float32
	for lane := 0; lane < 8; lane++ {
		v[lane] = (c[0][lane] + c[2][lane]) + (c[1][lane] + c[3][lane])
	}
	sum := (((v[0] + v[4]) + (v[1] + v[5])) + ((v[2] + v[6]) + (v[3] + v[7])))
	for ; i < n; i++ {
		sum += BF16ToF32(x[i]) * BF16ToF32(y[i])
	}
	return sum
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
		// Too few rows to benefit from parallelism
		for row := 0; row < rows; row++ {
			out[row] = BF16DotAsm(w[row*cols:(row+1)*cols], x)
		}
		return true
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
			for row := s; row < e; row++ {
				out[row] = BF16DotAsm(w[row*cols:(row+1)*cols], x)
			}
		}(start, end)
	}
	wg.Wait()
	return true
}
