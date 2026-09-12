package whisper

import (
	"context"
	"fmt"
	"math"
)

// Model-specific tensor/graph descriptions. These remain private so callers
// cannot mutate the generic Vulkan stages after operator admission.
type vkEncoderTensor struct {
	name  string
	shape []int
	data  []float32
	zero  bool
}
type vkEncoderStep struct {
	op, out       string
	in            []string
	stride        int
	channelsFirst bool
}
type vkEncoderLayout struct {
	cfg          Config
	frames, rows int
	weights      [][]vkEncoderTensor
	scratch      []vkEncoderTensor
	plans        [][]vkEncoderStep
}

// Preflight all shapes/data before any native resource is allocated. The source
// encoder must not be concurrently mutated during construction. No source slice
// is retained by the completed Vulkan encoder.
func describeVulkanEncoder(ctx context.Context, enc *Encoder, frames int) (*vkEncoderLayout, error) {
	if ctx == nil {
		return nil, fmt.Errorf("whisper Vulkan: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if enc == nil {
		return nil, fmt.Errorf("whisper Vulkan: nil encoder")
	}
	c := enc.cfg
	d, f := c.EncoderDModel, c.EncoderFFNDim
	if c.MaxLength < 2 || c.MaxLength > 4096 || c.MaxLength%2 != 0 || frames < 1 || frames > c.MaxLength || c.NumMelBins < 1 || c.NumMelBins > 2048 || d < 1 || d > 2048 || f < 1 || f > 16384 || c.EncoderLayers < 1 || c.EncoderLayers > 32 || len(enc.Layers) != c.EncoderLayers || c.EncoderHeads < 1 || c.EncoderHeads > 32 || c.HeadDim < 1 || c.HeadDim > 64 || c.HeadDim*c.EncoderHeads != d {
		return nil, fmt.Errorf("whisper Vulkan: unsupported encoder geometry")
	}
	l := &vkEncoderLayout{cfg: c, frames: frames, rows: (frames + 1) / 2}
	var bad error
	tensor := func(name string, data []float32, zero bool, shape ...int) vkEncoderTensor {
		n := 1
		for _, x := range shape {
			n *= x
		} // Fixed envelope above bounds all products.
		if zero && len(data) == 0 {
			return vkEncoderTensor{name, shape, nil, true}
		}
		if len(data) != n {
			bad = fmt.Errorf("whisper Vulkan: %s length %d, want %d", name, len(data), n)
		}
		if bad == nil {
			for i, v := range data {
				if i%16384 == 0 {
					if err := ctx.Err(); err != nil {
						bad = err
						break
					}
				}
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					bad = fmt.Errorf("whisper Vulkan: nonfinite %s[%d]", name, i)
					break
				}
			}
		}
		return vkEncoderTensor{name, shape, data, false}
	}
	stem := []vkEncoderTensor{tensor("conv1.w", enc.Conv1Weight, false, d, c.NumMelBins, 3), tensor("conv1.b", enc.Conv1Bias, false, d), tensor("conv2.w", enc.Conv2Weight, false, d, d, 3), tensor("conv2.b", enc.Conv2Bias, false, d), tensor("final.w", enc.FinalLNWeight, false, d), tensor("final.b", enc.FinalLNBias, false, d)}
	// Positional table has the complete configured extent; upload only this
	// fixed-length encoder's prefix, including ceil(T/2) for odd input lengths.
	pos := tensor("pos", enc.PosEmbed, false, c.MaxLength/2, d)
	if bad != nil {
		return nil, bad
	}
	pos.shape = []int{l.rows, d}
	pos.data = pos.data[:l.rows*d]
	stem = append(stem, pos)
	l.weights = append(l.weights, stem)
	for i := range enc.Layers {
		e := &enc.Layers[i]
		p := fmt.Sprintf("layer%d.", i)
		specs := []vkEncoderTensor{tensor(p+"attn.w", e.AttnLNWeight, false, d), tensor(p+"attn.b", e.AttnLNBias, false, d), tensor(p+"q.w", e.QWeight, false, d, d), tensor(p+"q.b", e.QBias, false, d), tensor(p+"k.w", e.KWeight, false, d, d), tensor(p+"k.b", e.KBias, true, d), tensor(p+"v.w", e.VWeight, false, d, d), tensor(p+"v.b", e.VBias, false, d), tensor(p+"o.w", e.OWeight, false, d, d), tensor(p+"o.b", e.OBias, false, d), tensor(p+"mlp.w", e.MLPLNWeight, false, d), tensor(p+"mlp.b", e.MLPLNBias, false, d), tensor(p+"fc1.w", e.FC1Weight, false, f, d), tensor(p+"fc1.b", e.FC1Bias, false, f), tensor(p+"fc2.w", e.FC2Weight, false, d, f), tensor(p+"fc2.b", e.FC2Bias, false, d)}
		if bad != nil {
			return nil, bad
		}
		l.weights = append(l.weights, specs)
	}
	scratch := func(name string, shape ...int) {
		l.scratch = append(l.scratch, vkEncoderTensor{name: name, shape: shape})
	}
	scratch("mel", c.NumMelBins, frames)
	scratch("stem", frames, d)
	for _, name := range []string{"h", "norm", "q", "k", "v", "tmp"} {
		scratch(name, l.rows, d)
	}
	scratch("ff", l.rows, f)
	step := func(op, out string, in ...string) vkEncoderStep { return vkEncoderStep{op: op, out: out, in: in} }
	l.plans = append(l.plans, []vkEncoderStep{{op: "conv", out: "stem", in: []string{"mel", "conv1.w", "conv1.b"}, stride: 1, channelsFirst: true}, step("gelu", "stem", "stem"), {op: "conv", out: "h", in: []string{"stem", "conv2.w", "conv2.b"}, stride: 2}, step("gelu", "h", "h"), step("add", "h", "h", "pos")})
	for i := range enc.Layers {
		p := fmt.Sprintf("layer%d.", i)
		l.plans = append(l.plans, []vkEncoderStep{
			step("norm", "norm", "h", p+"attn.w", p+"attn.b"), step("linear", "q", "norm", p+"q.w", p+"q.b"), step("linear", "k", "norm", p+"k.w", p+"k.b"), step("linear", "v", "norm", p+"v.w", p+"v.b"), step("attention", "norm", "q", "k", "v"), step("linear", "tmp", "norm", p+"o.w", p+"o.b"), step("add", "h", "h", "tmp"), step("norm", "norm", "h", p+"mlp.w", p+"mlp.b"), step("linear", "ff", "norm", p+"fc1.w", p+"fc1.b"), step("gelu", "ff", "ff"), step("linear", "tmp", "ff", p+"fc2.w", p+"fc2.b"), step("add", "h", "h", "tmp"),
		})
	}
	l.plans = append(l.plans, []vkEncoderStep{step("norm", "h", "h", "final.w", "final.b")})
	return l, ctx.Err()
}
func vkEncoderArenaBytes(specs []vkEncoderTensor, alignment uint64) (uint64, error) {
	if alignment < 4 {
		alignment = 4
	}
	if alignment&(alignment-1) != 0 {
		return 0, fmt.Errorf("whisper Vulkan: invalid arena alignment")
	}
	var size uint64
	for _, s := range specs {
		n := uint64(4)
		for _, d := range s.shape {
			if d < 1 || n > math.MaxUint64/uint64(d) {
				return 0, fmt.Errorf("whisper Vulkan: arena size overflow")
			}
			n *= uint64(d)
		}
		if size > math.MaxUint64-(alignment-1) {
			return 0, fmt.Errorf("whisper Vulkan: arena alignment overflow")
		}
		size = (size + alignment - 1) &^ (alignment - 1)
		if n > math.MaxUint64-size {
			return 0, fmt.Errorf("whisper Vulkan: arena extent overflow")
		}
		size += n
	}
	return size, nil
}
