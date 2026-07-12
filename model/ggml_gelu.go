package model

import "github.com/rcarmo/go-pherence/internal/ggmlfp16"

// ggmlGELUF32 matches ggml-cpu's GGML_UNARY_OP_GELU fast path when
// GGML_GELU_FP16 is enabled: inputs in [-10,10] are rounded to FP16 before the
// tanh-GELU table lookup and the table value itself is stored as FP16.
func ggmlGELUF32(x float32) float32 {
	return ggmlfp16.GELUFP16Lookup(x)
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
