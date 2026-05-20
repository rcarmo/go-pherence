package simd

import "math"

// LayerNormLastAxisTo writes layer normalization over a row-major [rows, cols]
// matrix into out. gamma and beta must either both be nil or both contain at
// least cols values. Inputs are validated before writing.
func LayerNormLastAxisTo(out, x []float32, rows, cols int, gamma, beta []float32, eps float32) bool {
	total, ok := checkedMulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(out) < total || len(x) < total {
		return false
	}
	if (gamma == nil) != (beta == nil) {
		return false
	}
	if gamma != nil && (len(gamma) < cols || len(beta) < cols) {
		return false
	}
	for r := 0; r < rows; r++ {
		off := r * cols
		row := x[off : off+cols]
		mean := float32(0)
		for _, v := range row {
			mean += v
		}
		mean /= float32(cols)
		variance := float32(0)
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(cols)
		stdInv := float32(1.0 / math.Sqrt(float64(variance+eps)))
		for c := 0; c < cols; c++ {
			v := (row[c] - mean) * stdInv
			if gamma != nil {
				v = gamma[c]*v + beta[c]
			}
			out[off+c] = v
		}
	}
	return true
}
