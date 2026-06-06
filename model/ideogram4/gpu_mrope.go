package ideogram4

import (
	"fmt"
	"os"
	"strings"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func gpuMRoPEEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_MROPE")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuMRoPEStrict() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_MROPE_STRICT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func applyMRoPEGPU(x []float32, rope *MRoPE, tokens, heads, headDim int) error {
	if rope == nil || rope.tokens != tokens || rope.headDim != headDim {
		return fmt.Errorf("invalid Ideogram4 GPU MRoPE tables")
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	return nvidia.IdeogramMRoPE(x, rope.cos, rope.sin, tokens, heads, headDim)
}
