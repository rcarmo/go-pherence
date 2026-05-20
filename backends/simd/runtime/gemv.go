package simd

// GemvRows computes out[rows] = W[rows,cols] · x[cols], where W is row-major.
// It is the backend-owned dense F32 GEMV reference used by CPU fallbacks and
// future optimized dispatch paths.
func GemvRows(out, x, w []float32, rows, cols int) bool {
	weightLen, ok := checkedMulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(out) < rows || len(x) < cols || len(w) < weightLen {
		return false
	}
	x = x[:cols]
	for row := 0; row < rows; row++ {
		out[row] = Sdot(x, w[row*cols:(row+1)*cols])
	}
	return true
}

// GemvCols computes out[cols] = x[rows] · W[rows,cols], where W is row-major.
func GemvCols(out, x, w []float32, rows, cols int) bool {
	weightLen, ok := checkedMulInt(rows, cols)
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
