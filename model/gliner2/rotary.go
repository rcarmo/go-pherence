package gliner2

import (
	"fmt"
	"math"
)

// RotaryBoundary rotates interleaved even/odd endpoint pairs. This differs from
// split-half decoder RoPE; reusing that kernel would silently change scores.
// Output is independent of the input and positions are boundary indices.
func RotaryBoundary(states [][]float32, positions []int, base float32) ([][]float32, error) {
	if len(states) != len(positions) || len(states) == 0 {
		return nil, fmt.Errorf("rotary boundary row mismatch or empty states")
	}
	dim := len(states[0])
	if dim == 0 || dim%2 != 0 || base <= 0 || math.IsNaN(float64(base)) || math.IsInf(float64(base), 0) {
		return nil, fmt.Errorf("invalid rotary dimension/base")
	}
	inv := make([]float32, dim/2)
	for i := range inv {
		inv[i] = 1 / float32(math.Pow(float64(base), float64(float32(2*i)/float32(dim))))
	}
	out := make([][]float32, len(states))
	for row, x := range states {
		if len(x) != dim {
			return nil, fmt.Errorf("rotary row %d dimension mismatch", row)
		}
		out[row] = make([]float32, dim)
		for i, f := range inv {
			angle := float32(positions[row]) * f
			c, s := float32(math.Cos(float64(angle))), float32(math.Sin(float64(angle)))
			even, odd := x[2*i], x[2*i+1]
			out[row][2*i] = even*c - odd*s
			out[row][2*i+1] = even*s + odd*c
		}
	}
	return out, nil
}
