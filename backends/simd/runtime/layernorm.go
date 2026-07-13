package simd

import (
	"github.com/rcarmo/go-pherence/internal/checked"
	"math"
)

// LayerNormLastAxisTo writes layer normalization over a row-major [rows, cols]
// matrix into out. gamma and beta must either both be nil or both contain at
// least cols values. Inputs are validated before writing.
func LayerNormLastAxisTo(out, x []float32, rows, cols int, gamma, beta []float32, eps float32) bool {
	total, ok := checked.MulInt(rows, cols)
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
		rowOut := out[off : off+cols]
		row := x[off : off+cols]
		if gamma != nil {
			layerNormAffineRowTo(rowOut, row, gamma[:cols], beta[:cols], eps)
		} else {
			layerNormAffineRowGo(rowOut, row, nil, nil, eps)
		}
	}
	return true
}

func layerNormAffineRowGo(out, x, gamma, beta []float32, eps float32) {
	mean := float32(0)
	for _, v := range x {
		mean += v
	}
	mean /= float32(len(x))
	variance := float32(0)
	for _, v := range x {
		d := v - mean
		variance += d * d
	}
	variance /= float32(len(x))
	stdInv := float32(1.0 / math.Sqrt(float64(variance+eps)))
	for i, input := range x {
		v := (input - mean) * stdInv
		if gamma != nil {
			v = gamma[i]*v + beta[i]
		}
		out[i] = v
	}
}
