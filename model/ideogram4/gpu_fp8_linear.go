package ideogram4

import (
	"fmt"
	"os"
	"strings"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// gpuFP8Enabled gates the first correctness-oriented Ideogram GPU path. It is
// deliberately opt-in because the current implementation streams one FP8 linear
// at a time to the GPU; this proves the CUDA kernel boundary without promising
// that the whole Ideogram graph is performance-ready or GPU-resident.
func gpuFP8Enabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_FP8")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuFP8Strict() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_FP8_STRICT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func (f *FP8Linear) applyGPUStreaming(x []float32, out []float32) error {
	if f == nil {
		return ErrRuntimeNotImplemented
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	w, err := nvidia.UploadFP8E4M3Linear(f.weight.Weight, f.weight.Scale, f.weight.Bias, f.weight.OutDim, f.weight.InDim)
	if err != nil {
		return err
	}
	defer w.Free()
	return nvidia.GemvFP8E4M3(out, x, w)
}
