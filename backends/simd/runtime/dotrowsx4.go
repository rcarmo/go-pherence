package simd

import "github.com/rcarmo/go-pherence/internal/checked"

// DotRowsx4 computes four dot products between one shared activation vector x
// and four consecutive F32 weight rows. w must contain at least 4*cols values
// laid out as [row0 row1 row2 row3], each row having cols values.
func DotRowsx4(w []float32, x []float32, cols int) (float32, float32, float32, float32, bool) {
	weightLen, ok := checked.MulInt(4, cols)
	if cols <= 0 || !ok || len(w) < weightLen || len(x) < cols {
		return 0, 0, 0, 0, false
	}
	d0, d1, d2, d3 := dotRowsx4(w[:weightLen], x[:cols], cols)
	return d0, d1, d2, d3, true
}

func dotRowsx4Scalar(w []float32, x []float32, cols int) (float32, float32, float32, float32) {
	x = x[:cols]
	return Sdot(x, w[:cols]),
		Sdot(x, w[cols:2*cols]),
		Sdot(x, w[2*cols:3*cols]),
		Sdot(x, w[3*cols:4*cols])
}
