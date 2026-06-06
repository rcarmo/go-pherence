package ideogram4

import (
	"fmt"

	"github.com/rcarmo/go-pherence/backends/simd/quant/fp8"
)

// FP8Linear is the concrete FP8LinearWeight backed by the SIMD fp8 backend.
// It pairs an Ideogram4 LinearSpec with a loaded E4M3 weight + scale tensor and
// exposes on-the-fly dequant GEMV without materializing F32 weights.
type FP8Linear struct {
	spec   LinearSpec
	weight fp8.Linear
	gpu    fp8LinearGPUCache
}

// NewFP8Linear binds raw E4M3 weight bytes and a scale tensor to a LinearSpec,
// validating that the byte/scale shapes match the spec's expected dimensions.
// bias may be nil.
func NewFP8Linear(spec LinearSpec, weightBytes []byte, scale []float32, bias []float32) (*FP8Linear, error) {
	if spec.OutDim <= 0 || spec.InDim <= 0 {
		return nil, fmt.Errorf("ideogram4 fp8 linear %q invalid dims out=%d in=%d", spec.Prefix, spec.OutDim, spec.InDim)
	}
	w := fp8.Linear{OutDim: spec.OutDim, InDim: spec.InDim, Weight: weightBytes, Scale: scale, Bias: bias}
	if err := w.Validate(); err != nil {
		return nil, fmt.Errorf("ideogram4 fp8 linear %q: %w", spec.Prefix, err)
	}
	return &FP8Linear{spec: spec, weight: w}, nil
}

func (f *FP8Linear) Role() LinearRole { return f.spec.Role }
func (f *FP8Linear) InDim() int       { return f.spec.InDim }
func (f *FP8Linear) OutDim() int      { return f.spec.OutDim }
func (f *FP8Linear) Spec() LinearSpec { return f.spec }

// ReleaseGPU frees any cached GPU-resident copy owned by this linear. It is a
// no-op unless GO_PHERENCE_IDEOGRAM4_GPU_FP8_CACHE has caused a cached upload.
func (f *FP8Linear) ReleaseGPU() {
	if f != nil {
		f.gpu.release()
	}
}

// Apply computes out = W*x using on-the-fly E4M3 dequant. By default this is
// the CPU/SIMD fp8 backend. When GO_PHERENCE_IDEOGRAM4_GPU_FP8=1 is set, Apply
// first tries the correctness-oriented NVIDIA GEMV path and falls back to CPU
// unless GO_PHERENCE_IDEOGRAM4_GPU_FP8_STRICT=1 is also set. Set
// GO_PHERENCE_IDEOGRAM4_GPU_FP8_CACHE=1 to keep the uploaded FP8 weight
// resident across calls instead of streaming it every projection.
func (f *FP8Linear) Apply(x []float32, out []float32) error {
	if f == nil {
		return ErrRuntimeNotImplemented
	}
	if gpuFP8Enabled() {
		var err error
		if gpuFP8CacheEnabled() {
			err = f.applyGPUCached(x, out)
		} else {
			err = f.applyGPUStreaming(x, out)
		}
		if err == nil || gpuFP8Strict() {
			return err
		}
	}
	return f.weight.GemvTo(x, out)
}

var _ FP8LinearWeight = (*FP8Linear)(nil)
