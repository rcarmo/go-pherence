package ideogram4

import (
	"fmt"
	"os"
	"strings"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func gpuAttentionEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_ATTN")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuAttentionStrict() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_ATTN_STRICT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func fullAttentionGPU(out, q, k, v []float32, tokens, heads, headDim int, scale float32) error {
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	return nvidia.IdeogramFullAttention(out, q, k, v, tokens, heads, headDim, scale)
}
