package ideogram4

import (
	"fmt"
	"os"
	"strings"
	"sync"

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

func gpuFP8CacheEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_FP8_CACHE")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

type fp8LinearGPUCache struct {
	mu     sync.Mutex
	weight *nvidia.GPUFP8E4M3Linear
	outDim int
	inDim  int
}

func (c *fp8LinearGPUCache) release() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.weight != nil {
		c.weight.Free()
		c.weight = nil
	}
	c.outDim = 0
	c.inDim = 0
}

func (f *FP8Linear) applyGPUCached(x []float32, out []float32) error {
	if f == nil {
		return ErrRuntimeNotImplemented
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable")
	}
	f.gpu.mu.Lock()
	defer f.gpu.mu.Unlock()
	if f.gpu.weight == nil || f.gpu.outDim != f.weight.OutDim || f.gpu.inDim != f.weight.InDim {
		if f.gpu.weight != nil {
			f.gpu.weight.Free()
			f.gpu.weight = nil
		}
		w, err := nvidia.UploadFP8E4M3Linear(f.weight.Weight, f.weight.Scale, f.weight.Bias, f.weight.OutDim, f.weight.InDim)
		if err != nil {
			return err
		}
		f.gpu.weight = w
		f.gpu.outDim = f.weight.OutDim
		f.gpu.inDim = f.weight.InDim
	}
	if err := nvidia.GemvFP8E4M3(out, x, f.gpu.weight); err != nil {
		f.gpu.weight.Free()
		f.gpu.weight = nil
		return err
	}
	return nil
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
