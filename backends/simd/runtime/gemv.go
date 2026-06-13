package simd

import "github.com/rcarmo/go-pherence/internal/checked"

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
//
// For batch > 1 this is deliberately not a loop of independent GEMV calls: it
// streams each W row once and accumulates all batch rows before moving to the
// next output row. DiffusionGemma uses this for selected MoE experts, e.g.
// [16,2816] × [704,2816]^T -> [16,704], where reusing the expert row across
// the 16 canvas positions avoids repeated weight walks.
func GemmRows(out, x, w []float32, batch, rows, cols int) bool {
	xLen, okX := checked.MulInt(batch, cols)
	outLen, okOut := checked.MulInt(batch, rows)
	weightLen, okW := checked.MulInt(rows, cols)
	if batch <= 0 || rows <= 0 || cols <= 0 || !okX || !okOut || !okW || len(out) < outLen || len(x) < xLen || len(w) < weightLen {
		return false
	}
	if batch == 1 {
		return GemvRows(out[:rows], x[:cols], w[:weightLen], rows, cols)
	}
	out = out[:outLen]
	x = x[:xLen]
	w = w[:weightLen]
	for i := range out {
		out[i] = 0
	}
	for row := 0; row < rows; row++ {
		wrow := w[row*cols : (row+1)*cols]
		for col, wc := range wrow {
			for b := 0; b < batch; b++ {
				out[b*rows+row] += x[b*cols+col] * wc
			}
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
