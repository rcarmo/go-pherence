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
	if gpuDisabledByK3() {
		return false
	}
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_FP8")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuFP8Strict() bool {
	if gpuDisabledByK3() {
		return false
	}
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_IDEOGRAM4_GPU_FP8_STRICT")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func gpuFP8CacheEnabled() bool {
	if gpuDisabledByK3() {
		return false
	}
	if gpuResidencyPolicy() == gpuResidencyStream {
		return false
	}
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
		return fmt.Errorf("nvidia runtime unavailable: fp8")
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
		return fmt.Errorf("nvidia runtime unavailable: fp8")
	}
	w, err := nvidia.UploadFP8E4M3Linear(f.weight.Weight, f.weight.Scale, f.weight.Bias, f.weight.OutDim, f.weight.InDim)
	if err != nil {
		return err
	}
	defer w.Free()
	return nvidia.GemvFP8E4M3(out, x, w)
}

func (f *FP8Linear) applyGPUBatchCached(x []float32, out []float32, batch int) error {
	if f == nil {
		return ErrRuntimeNotImplemented
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: fp8")
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
	if err := nvidia.GemmFP8E4M3(out, x, batch, f.gpu.weight); err != nil {
		f.gpu.weight.Free()
		f.gpu.weight = nil
		return err
	}
	return nil
}

func (f *FP8Linear) applyGPUBatchStreaming(x []float32, out []float32, batch int) error {
	if f == nil {
		return ErrRuntimeNotImplemented
	}
	if !nvidia.Available() {
		return fmt.Errorf("nvidia runtime unavailable: fp8")
	}
	w, err := nvidia.UploadFP8E4M3Linear(f.weight.Weight, f.weight.Scale, f.weight.Bias, f.weight.OutDim, f.weight.InDim)
	if err != nil {
		return err
	}
	defer w.Free()
	return nvidia.GemmFP8E4M3(out, x, batch, w)
}

func applyBatch2SameInput(a, b *FP8Linear, x, outA, outB []float32, batch int) error {
	if a == nil || b == nil {
		return ErrRuntimeNotImplemented
	}
	if gpuFP8Enabled() && nvidia.Available() {
		if gpuFP8CacheEnabled() {
			a.gpu.mu.Lock()
			defer a.gpu.mu.Unlock()
			b.gpu.mu.Lock()
			defer b.gpu.mu.Unlock()
			if err := ensureFP8LinearGPUCachedLocked(a); err != nil {
				if gpuFP8Strict() {
					return err
				}
			} else if err := ensureFP8LinearGPUCachedLocked(b); err != nil {
				if gpuFP8Strict() {
					return err
				}
			} else if err := nvidia.Gemm2FP8E4M3SameInput(outA, outB, x, batch, a.gpu.weight, b.gpu.weight); err == nil || gpuFP8Strict() {
				return err
			}
		} else {
			wa, err := nvidia.UploadFP8E4M3Linear(a.weight.Weight, a.weight.Scale, a.weight.Bias, a.weight.OutDim, a.weight.InDim)
			if err != nil {
				if gpuFP8Strict() {
					return err
				}
			} else {
				defer wa.Free()
				wb, err := nvidia.UploadFP8E4M3Linear(b.weight.Weight, b.weight.Scale, b.weight.Bias, b.weight.OutDim, b.weight.InDim)
				if err != nil {
					if gpuFP8Strict() {
						return err
					}
				} else {
					defer wb.Free()
					if err := nvidia.Gemm2FP8E4M3SameInput(outA, outB, x, batch, wa, wb); err == nil || gpuFP8Strict() {
						return err
					}
				}
			}
		}
	}
	if err := a.ApplyBatch(x, outA, batch); err != nil {
		return err
	}
	return b.ApplyBatch(x, outB, batch)
}

func ensureFP8LinearGPUCachedLocked(f *FP8Linear) error {
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
	return nil
}
