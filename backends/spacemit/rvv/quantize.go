package rvv

import "math"

// QuantizeF32RowQ8 quantizes a float32 row into int8 with block-scale Q8_0 format.
// Returns the scale and negative sum for A-block layout. dst must have len >= 32.
// This is a scalar fast path; a future RVV assembly version can vectorize maxAbs + round.
func QuantizeF32RowQ8Block(src []float32, dst []int8) (scale float32, negSum int16) {
	maxAbs := float32(0)
	for _, v := range src {
		av := v
		if av < 0 {
			av = -av
		}
		if av > maxAbs {
			maxAbs = av
		}
	}
	if maxAbs == 0 {
		return 0, 0
	}
	scale = maxAbs / 127.0
	inv := 1.0 / scale
	sum := 0
	for i, v := range src {
		q := int(math.Round(float64(v * inv)))
		if q > 127 {
			q = 127
		}
		if q < -128 {
			q = -128
		}
		dst[i] = int8(q)
		sum += q
	}
	return scale, int16(-sum)
}
