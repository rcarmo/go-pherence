package vulkan

import (
	"context"
	"fmt"
	"math"
	"runtime"
	"unsafe"
)

// VkLayerNormF32 owns the row-wise LayerNorm pipeline. Forward/Stage use arena
// tensors directly with no intermediate host transfer. This initial operator
// uses centered population variance, learned scale+bias and F32 arithmetic.
// Static/mocked qualification is not device numerical or model qualification.
// Copies share the underlying kernel lifetime; Close invalidates derived plans.
type VkLayerNormF32 struct{ kernel *VkComputeKernel }

func NewVkLayerNormF32(ctx context.Context) (*VkLayerNormF32, error) {
	if err := vkAcquire(ctx); err != nil {
		return nil, err
	}
	defer vkRelease()
	// Native construction is not preemptible; cancellation is checked on entry.
	k, err := vkKernelCreateLocked(spirv_layer_norm_f32, 4, 12)
	if err != nil {
		return nil, err
	}
	return &VkLayerNormF32{kernel: k}, nil
}
func (op *VkLayerNormF32) Close() error {
	if op == nil {
		return nil
	}
	return op.kernel.Close()
}

// Stage returns owned descriptor/push slices for NewVkF32Plan. X/Out must have
// identical [rows,width] shapes, width1..16384. Weight/Bias are [width]. Exact
// X/Out alias is allowed; partial overlap and any output overlap with parameters
// is rejected. Read-only inputs may overlap. Epsilon must be positive finite.
// Contents must be finite and have representable F32 sums/variance: no tensor
// scan occurs here and overflow/driver rounding are not hidden by fallback.
// Like any VkF32Stage, callers may mutate it only under their own responsibility.
func (op *VkLayerNormF32) Stage(ctx context.Context, out, x, weight, bias *VkTensorF32, eps float32) (VkF32Stage, error) {
	if err := vkAcquire(ctx); err != nil {
		return VkF32Stage{}, err
	}
	defer vkRelease()
	return op.stageLocked(out, x, weight, bias, eps)
}
func (op *VkLayerNormF32) stageLocked(out, x, weight, bias *VkTensorF32, eps float32) (VkF32Stage, error) {
	fail := func(s string) (VkF32Stage, error) { return VkF32Stage{}, fmt.Errorf("Vulkan LayerNorm: %s", s) }
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
	if x.rank != 2 || out.rank != 2 || x.shape[0] != out.shape[0] || x.shape[1] != out.shape[1] {
		return fail("X and Out must have identical rank2 shapes")
	}
	rows, width := x.shape[0], x.shape[1]
	if rows < 1 || width < 1 || width > 16384 || uint64(rows) > uint64(^uint32(0))/uint64(width) {
		return fail("rows/width overflow or unsupported width")
	}
	if weight.rank != 1 || bias.rank != 1 || weight.shape[0] != width || bias.shape[0] != width {
		return fail("weight/bias must have width elements")
	}
	if math.IsNaN(float64(eps)) || math.IsInf(float64(eps), 0) || eps <= 0 {
		return fail("epsilon must be finite and positive")
	}
	if bindings[0].size != uint64(rows)*uint64(width)*4 || bindings[3].size != bindings[0].size || bindings[1].size != uint64(width)*4 || bindings[2].size != uint64(width)*4 {
		return fail("shape/storage size mismatch")
	}
	exact := bindings[0] == bindings[3]
	if (!exact && vkBindingsOverlap(bindings[0], bindings[3])) || vkBindingsOverlap(bindings[1], bindings[3]) || vkBindingsOverlap(bindings[2], bindings[3]) {
		return fail("unsafe output alias")
	}
	push := []uint32{uint32(rows), uint32(width), math.Float32bits(eps)}
	// Reuse core owner/function/dispatch-limit checks before exposing the stage.
	if err := op.kernel.validateBindingsLocked(uint32(rows), 1, 1, bindings, unsafePushWords(push)); err != nil {
		return VkF32Stage{}, err
	}
	return VkF32Stage{Kernel: op.kernel, Groups: [3]uint32{uint32(rows), 1, 1}, Tensors: tensors, PushWords: push}, nil
}

// Bindings already passed subtraction-based allocation range checks. Use
// differences here too, avoiding an overflow-prone end=start+size calculation.
func unsafePushWords(words []uint32) unsafe.Pointer {
	if len(words) == 0 {
		return nil
	}
	return unsafe.Pointer(&words[0])
}
func vkBindingsOverlap(a, b vkBufferBinding) bool {
	if a.buffer != b.buffer {
		return false
	}
	if a.offset <= b.offset {
		return b.offset-a.offset < a.size
	}
	return a.offset-b.offset < b.size
}

func (op *VkLayerNormF32) Forward(ctx context.Context, out, x, weight, bias *VkTensorF32, eps float32) error {
	if err := vkAcquire(ctx); err != nil {
		return err
	}
	defer vkRelease()
	stage, err := op.stageLocked(out, x, weight, bias, eps)
	if err != nil {
		return err
	}
	bindings := make([]vkBufferBinding, len(stage.Tensors))
	for i, t := range stage.Tensors {
		bindings[i], err = t.bindingLocked()
		if err != nil {
			return err
		}
	}
	err = op.kernel.dispatchBindingsLocked(ctx, stage.Groups[0], 1, 1, bindings, unsafePushWords(stage.PushWords))
	runtime.KeepAlive(stage.PushWords)
	return err
}
