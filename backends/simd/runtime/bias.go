package simd

import "github.com/rcarmo/go-pherence/internal/checked"

// AddBiasRowsTo adds bias[cols] to each row of dst[rows,cols] in-place.
func AddBiasRowsTo(dst, bias []float32, rows, cols int) bool {
	total, ok := checked.MulInt(rows, cols)
	if rows <= 0 || cols <= 0 || !ok || len(dst) < total || len(bias) < cols {
		return false
	}
	for r := 0; r < rows; r++ {
		row := dst[r*cols : (r+1)*cols]
		for c := 0; c < cols; c++ {
			row[c] += bias[c]
		}
	}
	return true
}
