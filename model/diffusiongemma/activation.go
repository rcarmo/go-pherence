package diffusiongemma

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

// diffusionGemmaGELUMulTo matches llama.cpp's ggml_gelu / LLM_FFN_GELU,
// which is the tanh-approximation GELU, not the erf GELU variant.
func diffusionGemmaGELUMulTo(dst, gate, up []float32) bool {
	return simd.GELUTanhMulTo(dst, gate, up)
}
