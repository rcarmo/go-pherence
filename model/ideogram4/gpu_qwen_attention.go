package ideogram4

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func qwenGQAAttentionGPU(out, qRow, kPrefix, vPrefix []float32, seqLen, heads, kvHeads, headDim int, scale float32) error {
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: qwen_attention")
	}
	return nvidia.F32GQAAttention(out, qRow, kPrefix, vPrefix, seqLen, heads, kvHeads, headDim, scale)
}
