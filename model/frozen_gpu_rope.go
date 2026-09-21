package model

import "math"

// Qwen's reference constructs inverse frequencies and position products in
// float32 before trig. Keep this policy private to the experimental encoder;
// the generic/GGML paths have different established rounding contracts.
func frozenQwenRoPE(tokens, headDim int, theta float64) []float32 {
	out := make([]float32, tokens*headDim)
	for i := 0; i < headDim/2; i++ {
		power := float32(math.Pow(float64(float32(theta)), float64(float32(2*i)/float32(headDim))))
		inv := float32(1) / power
		for pos := 0; pos < tokens; pos++ {
			a := float32(pos) * inv
			out[pos*headDim+i*2] = float32(math.Cos(float64(a)))
			out[pos*headDim+i*2+1] = float32(math.Sin(float64(a)))
		}
	}
	return out
}
