package vulkan

import (
	"context"
	"fmt"
	"math"
	"runtime"
)

// VkAttentionF32 is non-causal, equal-head F32 attention over time-major
// Q[seqQ,heads*headDim], K/V[seqKV,heads*headDim] and Out shaped like Q.
// A fused16-query/16-key tile uses online softmax without a quadratic score
// allocation. headDim is1..64, heads1..32, and both sequence lengths1..4096.
// No mask, GQA, F16, quantisation or model/default selection is provided.
// Caller contents must have finite representable dot products and weighted
// sums; no data scan. Device accuracy/performance still need qualification.
type VkAttentionF32 struct{ kernel *VkComputeKernel }

func NewVkAttentionF32(ctx context.Context) (*VkAttentionF32, error) {
	return newVkAttentionF32Code(ctx, spirv_attention_f32)
}

// NewVkAttentionKey32F32 explicitly selects 32-key online softmax with parallel
// exponent evaluation and F32 FMA. It changes reduction/rounding relative to
// the baseline and must be separately numerically/task-qualified. No fallback.
func NewVkAttentionKey32F32(ctx context.Context) (*VkAttentionF32, error) {
	return newVkAttentionF32Code(ctx, spirv_attention_f32_key32)
}

// NewVkAttentionKey32ScoreILPF32 explicitly interleaves two independent score
// accumulators while preserving increasing-channel FMA order for each key.
// The key32 online-softmax and output-reduction orders remain unchanged.
// This is an opt-in kernel, with no fallback or serving default selection.
func NewVkAttentionKey32ScoreILPF32(ctx context.Context) (*VkAttentionF32, error) {
	return newVkAttentionF32Code(ctx, spirv_attention_f32_key32_scoreilp)
}
func newVkAttentionF32Code(ctx context.Context, code []byte) (*VkAttentionF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	k, err := vkKernelCreateLocked(code, 4, 20)
	if err != nil {
		return nil, err
	}
	return &VkAttentionF32{kernel: k}, nil
}
func (op *VkAttentionF32) Close() error {
	if op == nil {
		return nil
	}
	return op.kernel.Close()
}

// Stage checks exact rank/shape/extents and rejects every output/input overlap.
// Read-only inputs may alias. Tensors and push words in returned stages are
// owned slices; modifying generic stages must not bypass these operator checks.
func (op *VkAttentionF32) Stage(ctx context.Context, out, q, k, v *VkTensorF32, heads int) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(out, q, k, v, heads)
}
func (op *VkAttentionF32) stageLocked(out, q, k, v *VkTensorF32, heads int) (VkF32Stage, error) {
	fail := func(s string) (VkF32Stage, error) { return VkF32Stage{}, fmt.Errorf("Vulkan AttentionF32: %s", s) }
	if op == nil || op.kernel == nil {
		return fail("uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	tensors := []*VkTensorF32{q, k, v, out}
	bindings := make([]vkBufferBinding, 4)
	for i, t := range tensors {
		b, err := t.bindingLocked()
		if err != nil {
			return VkF32Stage{}, err
		}
		bindings[i] = b
		if t.rank != 2 {
			return fail("rank must be2")
		}
	}
	seqQ, seqKV, width := q.shape[0], k.shape[0], q.shape[1]
	if seqQ < 1 || seqQ > 4096 || seqKV < 1 || seqKV > 4096 || heads < 1 || heads > 32 || width < 1 || width > 2048 || width%heads != 0 {
		return fail("sequence/head/width envelope")
	}
	dim := width / heads
	if dim < 1 || dim > 64 {
		return fail("head dimension must be1..64")
	}
	if k.shape[1] != width || v.shape[0] != seqKV || v.shape[1] != width || out.shape[0] != seqQ || out.shape[1] != width {
		return fail("attention shapes mismatch")
	}
	sizes := [4]uint64{uint64(seqQ) * uint64(width) * 4, uint64(seqKV) * uint64(width) * 4, uint64(seqKV) * uint64(width) * 4, uint64(seqQ) * uint64(width) * 4}
	for i, b := range bindings {
		if b.size != sizes[i] {
			return fail("shape/storage size mismatch")
		}
		if i < 3 && vkBindingsOverlap(b, bindings[3]) {
			return fail("output overlaps input")
		}
	}
	groups := [3]uint32{uint32((seqQ + 15) / 16), uint32(heads), 1}
	push := []uint32{uint32(seqQ), uint32(seqKV), uint32(heads), uint32(dim), math.Float32bits(float32(1 / math.Sqrt(float64(dim))))}
	if err := op.kernel.validateBindingsLocked(groups[0], groups[1], groups[2], bindings, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: groups, Tensors: tensors, PushWords: push}, nil
}
func (op *VkAttentionF32) Forward(ctx context.Context, out, q, k, v *VkTensorF32, heads int) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	stage, err := op.stageLocked(out, q, k, v, heads)
	if err != nil {
		return err
	}
	bindings := make([]vkBufferBinding, 4)
	for i, t := range stage.Tensors {
		bindings[i], err = t.bindingLocked()
		if err != nil {
			return err
		}
	}
	err = op.kernel.dispatchBindingsLocked(ctx, stage.Groups[0], stage.Groups[1], 1, bindings, unsafePushWords(stage.PushWords))
	runtime.KeepAlive(stage.PushWords)
	return err
}
