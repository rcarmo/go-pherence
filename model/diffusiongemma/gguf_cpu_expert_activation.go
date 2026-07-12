package diffusiongemma

import "github.com/rcarmo/go-pherence/internal/ggmlfp16"

func ggufExpertGGMLGELUFP16MulTo(dst, gate, up []float32) bool {
	return ggmlfp16.GELUFP16LookupMulTo(dst, gate, up)
}
