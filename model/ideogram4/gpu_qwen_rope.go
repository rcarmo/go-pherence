package ideogram4

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func qwenRoPEGPU(vec []float32, rt ropeTable, t int) error {
	if len(vec) < rt.headDim || rt.headDim <= 0 || rt.half <= 0 || t < 0 {
		return fmt.Errorf("invalid Qwen RoPE row vec=%d headDim=%d half=%d pos=%d", len(vec), rt.headDim, rt.half, t)
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	return nvidia.F32RoPE(vec[:rt.headDim], rt.cos, rt.sin, t, 1, rt.headDim)
}
