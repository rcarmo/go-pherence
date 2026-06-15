package diffusiongemma

import (
	"os"
	"strings"

	simdfp8 "github.com/rcarmo/go-pherence/backends/simd/quant/fp8"
)

func diffusionGemmaFP8DynamicActivationEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_FP8_DYNAMIC_ACT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func quantizeDynamicTokenBatch(dst, src []float32, batch, dim int) []float32 {
	need := batch * dim
	if batch <= 0 || dim <= 0 || len(src) < need {
		return src
	}
	if len(dst) < need {
		dst = make([]float32, need)
	}
	for b := 0; b < batch; b++ {
		simdfp8.QuantizeTokenE4M3DequantTo(dst[b*dim:(b+1)*dim], src[b*dim:(b+1)*dim])
	}
	return dst[:need]
}

func quantizeDynamicTokenRow(dst, src []float32) []float32 {
	if len(dst) < len(src) {
		dst = make([]float32, len(src))
	}
	simdfp8.QuantizeTokenE4M3DequantTo(dst[:len(src)], src)
	return dst[:len(src)]
}
