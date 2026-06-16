package model

import (
	"math"

	"github.com/rcarmo/go-pherence/half"
)

// ggmlGELUF32 matches ggml-cpu's GGML_UNARY_OP_GELU fast path when
// GGML_GELU_FP16 is enabled: inputs in [-10,10] are rounded to FP16 before the
// tanh-GELU table lookup and the table value itself is stored as FP16.
func ggmlGELUF32(x float32) float32 {
	if x <= -10.0 {
		return 0
	}
	if x >= 10.0 {
		return x
	}
	xh := half.F16ToF32(half.F32ToF16(x))
	const sqrt2OverPi = float32(0.79788456080286535587989211986876)
	inner := sqrt2OverPi * xh * (1 + 0.044715*xh*xh)
	g := 0.5 * xh * (1 + float32(math.Tanh(float64(inner))))
	return half.F16ToF32(half.F32ToF16(g))
}

func ggmlGELUMulInPlace(gate, up []float32) {
	n := len(gate)
	if len(up) < n {
		n = len(up)
	}
	for i := 0; i < n; i++ {
		gate[i] = ggmlGELUF32(gate[i]) * up[i]
	}
}
