package simd

import "github.com/rcarmo/go-pherence/internal/checked"

// BF16DotF32x4 computes four dot products between four consecutive BF16 weight
// rows and one shared F32 activation vector. w must contain at least 4*cols
// BF16 values laid out as [row0 row1 row2 row3], each row having cols values.
func BF16DotF32x4(w []uint16, x []float32, cols int) (float32, float32, float32, float32, bool) {
	weightLen, ok := checked.MulInt(4, cols)
	if cols <= 0 || !ok || len(w) < weightLen || len(x) < cols {
		return 0, 0, 0, 0, false
	}
	d0, d1, d2, d3 := bf16DotF32x4(w[:weightLen], x[:cols], cols)
	return d0, d1, d2, d3, true
}

// BF16DotBF16x4 computes four dot products between four consecutive BF16
// weight rows and one shared BF16 activation vector. w must contain at least
// 4*cols BF16 values laid out as [row0 row1 row2 row3].
func BF16DotBF16x4(w []uint16, x []uint16, cols int) (float32, float32, float32, float32, bool) {
	weightLen, ok := checked.MulInt(4, cols)
	if cols <= 0 || !ok || len(w) < weightLen || len(x) < cols {
		return 0, 0, 0, 0, false
	}
	d0, d1, d2, d3 := bf16DotBF16x4(w[:weightLen], x[:cols], cols)
	return d0, d1, d2, d3, true
}

func bf16DotF32x4Scalar(w []uint16, x []float32, cols int) (float32, float32, float32, float32) {
	return BF16DotF32(w[:cols], x[:cols]),
		BF16DotF32(w[cols:2*cols], x[:cols]),
		BF16DotF32(w[2*cols:3*cols], x[:cols]),
		BF16DotF32(w[3*cols:4*cols], x[:cols])
}

func bf16DotBF16x4Scalar(w []uint16, x []uint16, cols int) (float32, float32, float32, float32) {
	return BF16Dot(w[:cols], x[:cols]),
		BF16Dot(w[cols:2*cols], x[:cols]),
		BF16Dot(w[2*cols:3*cols], x[:cols]),
		BF16Dot(w[3*cols:4*cols], x[:cols])
}
