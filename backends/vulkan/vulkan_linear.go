package vulkan

import (
	"context"
	"fmt"
	"runtime"
)

// VkLinearF32 owns a16x16 tiled F32 projection kernel: X[M,K]*W[N,K]^T+B[N].
// No quantised interpretation, packing or transpose copy. Static and source-
// model tests are not device numerical/performance qualification. Stage/Forward
// use resident arena tensors; caller owns weights/bias/activations. Copies share
// kernel lifetime and Close invalidates stages/plans referencing the operator.
type VkLinearF32 struct{ kernel *VkComputeKernel }

func NewVkLinearF32(ctx context.Context) (*VkLinearF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	k, err := vkKernelCreateLocked(spirv_linear_f32, 4, 12)
	if err != nil {
		return nil, err
	}
	return &VkLinearF32{kernel: k}, nil
}
func (op *VkLinearF32) Close() error {
	if op == nil {
		return nil
	}
	return op.kernel.Close()
}

// Stage validates fixed contiguous shapes: X[M,K],Weight[N,K],Bias[N],Out[M,N].
// Positive M/N/K<=16384, plus queried resource limits, bound all shader indexing
// and per-invocation loops. Output may NOT overlap any input, even exact alias.
// Inputs may overlap each other. Supply a zero bias tensor for no-bias projection.
// No finite-value scan: caller must ensure representable F32 sums/products.
func (op *VkLinearF32) Stage(ctx context.Context, out, x, weight, bias *VkTensorF32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(out, x, weight, bias)
}
func (op *VkLinearF32) stageLocked(out, x, weight, bias *VkTensorF32) (VkF32Stage, error) {
	fail := func(s string) (VkF32Stage, error) { return VkF32Stage{}, fmt.Errorf("Vulkan LinearF32: %s", s) }
	if op == nil || op.kernel == nil {
		return fail("uninitialized operator")
	}
	if err := vkStatusLocked(); err != nil {
		return VkF32Stage{}, err
	}
	tensors := []*VkTensorF32{x, weight, bias, out}
	bindings := make([]vkBufferBinding, 4)
	for i, t := range tensors {
		b, err := t.bindingLocked()
		if err != nil {
			return VkF32Stage{}, err
		}
		bindings[i] = b
	}
	if x.rank != 2 || weight.rank != 2 || bias.rank != 1 || out.rank != 2 {
		return fail("rank mismatch")
	}
	rows, inDim, outDim := x.shape[0], x.shape[1], weight.shape[0]
	if rows < 1 || rows > 16384 || inDim < 1 || inDim > 16384 || outDim < 1 || outDim > 16384 {
		return fail("dimensions must be1..16384")
	}
	if weight.shape[1] != inDim || bias.shape[0] != outDim || out.shape[0] != rows || out.shape[1] != outDim {
		return fail("projection shapes mismatch")
	}
	// Products fit uint32 and hostint by the fixed per-axis envelope. Storage
	// extents still obey actual device descriptor limits, checked below.
	sizes := [4]uint64{uint64(rows) * uint64(inDim) * 4, uint64(outDim) * uint64(inDim) * 4, uint64(outDim) * 4, uint64(rows) * uint64(outDim) * 4}
	for i, b := range bindings {
		if b.size != sizes[i] {
			return fail("shape/storage size mismatch")
		}
		if i < 3 && vkBindingsOverlap(b, bindings[3]) {
			return fail("output overlaps input")
		}
	}
	groups := [3]uint32{uint32((outDim + 15) / 16), uint32((rows + 15) / 16), 1}
	push := []uint32{uint32(rows), uint32(inDim), uint32(outDim)}
	if err := op.kernel.validateBindingsLocked(groups[0], groups[1], groups[2], bindings, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: groups, Tensors: tensors, PushWords: push}, nil
}
func (op *VkLinearF32) Forward(ctx context.Context, out, x, weight, bias *VkTensorF32) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	stage, err := op.stageLocked(out, x, weight, bias)
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
	err = op.kernel.dispatchBindingsLocked(ctx, stage.Groups[0], stage.Groups[1], stage.Groups[2], bindings, unsafePushWords(stage.PushWords))
	runtime.KeepAlive(stage.PushWords)
	return err
}
