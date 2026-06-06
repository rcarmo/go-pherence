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

// Apply computes out = W*x using on-the-fly E4M3 dequant.
func (f *FP8Linear) Apply(x []float32, out []float32) error {
	if f == nil {
		return ErrRuntimeNotImplemented
	}
	return f.weight.GemvTo(x, out)
}

var _ FP8LinearWeight = (*FP8Linear)(nil)
