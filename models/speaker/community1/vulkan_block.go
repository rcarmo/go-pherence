package community1

import (
	"context"
	"errors"
	"fmt"
	"math"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// VulkanBasicBlock is an explicit fixed-shape F32 owner for one Community-1
// WeSpeaker BasicBlock. It owns uploaded convolution weights, prepared BatchNorm
// coefficients, scratch tensors, operators and one private plan. Copies share
// ownership and serialisation. No model/default selects it; callers must Close.
type VulkanBasicBlock struct{ s *vulkanBasicBlockState }

type VulkanBasicBlockStats struct {
	Input, Output             CHWShape
	Stages                    int
	WeightBytes, ScratchBytes uint64
}

type vulkanBlockCloser interface{ Close() error }

type vulkanBasicBlockState struct {
	gate             chan struct{}
	stopping, closed bool
	stats            VulkanBasicBlockStats
	resources        []vulkanBlockCloser
	plan             *vk.VkF32Plan
	input, output    *vk.VkTensorF32
	conv             *vk.VkConv2DCHWF32
	affine           *vk.VkChannelAffineReLUF32
	add              *vk.VkAddF32
}

// NewVulkanBasicBlock requires explicit prior VulkanInit. Complete CPU-owned
// weights and shape are validated and copied before native allocation. On
// failure, normal rollback returns nil; if cleanup fails, a nonnil stopping
// owner is returned so Close can be retried after VulkanDrain.
func NewVulkanBasicBlock(ctx context.Context, source *WeSpeakerBasicBlock, input CHWShape) (*VulkanBasicBlock, error) {
	return newVulkanBasicBlock(ctx, source, input, vk.NewVkF32Plan)
}

func newVulkanBasicBlock(ctx context.Context, source *WeSpeakerBasicBlock, input CHWShape, makePlan func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error)) (result *VulkanBasicBlock, err error) {
	if makePlan == nil {
		return nil, fmt.Errorf("Community-1 Vulkan block: nil plan constructor")
	}
	layout, err := describeVulkanBasicBlock(ctx, source, input)
	if err != nil {
		return nil, err
	}
	limits, err := vk.VulkanLimits()
	if err != nil {
		return nil, err
	}
	arenaBytes, err := vulkanBlockArenaBytes(layout.tensors, limits.StorageBufferOffsetAlignment)
	if err != nil {
		return nil, err
	}
	if arenaBytes > uint64(limits.StorageBufferRange) || arenaBytes > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("Community-1 Vulkan block: arena exceeds device range")
	}
	s := &vulkanBasicBlockState{gate: make(chan struct{}, 1), stats: VulkanBasicBlockStats{Input: layout.input, Output: layout.output, Stages: len(layout.steps), WeightBytes: layout.weightBytes, ScratchBytes: layout.scratchBytes}}
	owner := &VulkanBasicBlock{s: s}
	defer func() {
		if err != nil {
			if closeErr := owner.Close(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Community-1 Vulkan block: rollback: %w", closeErr))
			}
		}
	}()
	if s.conv, err = vk.NewVkConv2DCHWF32(ctx); err != nil {
		return nil, err
	}
	s.resources = append(s.resources, s.conv)
	if s.affine, err = vk.NewVkChannelAffineReLUF32(ctx); err != nil {
		return nil, err
	}
	s.resources = append(s.resources, s.affine)
	if s.add, err = vk.NewVkAddF32(ctx); err != nil {
		return nil, err
	}
	s.resources = append(s.resources, s.add)
	arena, err := vk.NewVkTensorArena(ctx, int(arenaBytes))
	if err != nil {
		return nil, err
	}
	s.resources = append(s.resources, arena)
	tensors := make(map[string]*vk.VkTensorF32, len(layout.tensors))
	for _, spec := range layout.tensors {
		tensor, allocErr := arena.AllocF32(ctx, spec.shape...)
		if allocErr != nil {
			return nil, allocErr
		}
		tensors[spec.name] = tensor
		if spec.data != nil {
			if err = tensor.Upload(ctx, spec.data); err != nil {
				return nil, err
			}
		}
	}
	s.input, s.output = tensors["input"], tensors["b"]
	stages := make([]vk.VkF32Stage, 0, len(layout.steps))
	for _, step := range layout.steps {
		var stage vk.VkF32Stage
		switch step.op {
		case "conv":
			stage, err = s.conv.Stage(ctx, tensors[step.out], tensors[step.x], tensors[step.a], step.kernel, step.stride, step.padding)
		case "affine":
			stage, err = s.affine.Stage(ctx, tensors[step.out], tensors[step.x], tensors[step.a], tensors[step.b], step.relu)
		case "add":
			stage, err = s.add.Stage(ctx, tensors[step.out], tensors[step.x], tensors[step.a])
		default:
			return nil, fmt.Errorf("Community-1 Vulkan block: unknown graph operator %q", step.op)
		}
		if err != nil {
			return nil, err
		}
		stages = append(stages, stage)
	}
	plan, err := makePlan(ctx, stages)
	if plan != nil {
		s.resources = append(s.resources, plan)
		s.plan = plan
	}
	if err != nil {
		return nil, err
	}
	if plan == nil {
		return nil, fmt.Errorf("Community-1 Vulkan block: plan constructor returned nil")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return owner, nil
}

func (b *VulkanBasicBlock) acquire(ctx context.Context) (*vulkanBasicBlockState, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan block: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if b == nil || b.s == nil {
		return nil, fmt.Errorf("Community-1 Vulkan block: uninitialized owner")
	}
	s := b.s
	select {
	case s.gate <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-s.gate
			return nil, err
		}
		return s, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (b *VulkanBasicBlock) Stats() VulkanBasicBlockStats {
	if b == nil || b.s == nil {
		return VulkanBasicBlockStats{}
	}
	return b.s.stats
}

// Forward uploads one CHW input, executes one resident plan and downloads its
// owned CHW output. Calls are serialized. Inputs/outputs are checked for finite
// values; a native dispatch already in flight follows VulkanDrain semantics.
func (b *VulkanBasicBlock) Forward(ctx context.Context, input []float32) ([]float32, CHWShape, error) {
	fail := func(err error) ([]float32, CHWShape, error) { return nil, CHWShape{}, err }
	s, err := b.acquire(ctx)
	if err != nil {
		return fail(err)
	}
	defer func() { <-s.gate }()
	if s.stopping || s.closed {
		return fail(vk.ErrVulkanClosed)
	}
	if len(input) != s.input.Elements() {
		return fail(fmt.Errorf("Community-1 Vulkan block: input length %d, want %d", len(input), s.input.Elements()))
	}
	for i, value := range input {
		if i%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return fail(err)
			}
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fail(fmt.Errorf("Community-1 Vulkan block: nonfinite input[%d]", i))
		}
	}
	if err := s.input.Upload(ctx, input); err != nil {
		return fail(err)
	}
	if err := s.plan.Run(ctx); err != nil {
		return fail(fmt.Errorf("Community-1 Vulkan block: plan: %w", err))
	}
	output := make([]float32, s.output.Elements())
	if err := s.output.Download(ctx, output); err != nil {
		return fail(err)
	}
	for i, value := range output {
		if i%16384 == 0 {
			if err := ctx.Err(); err != nil {
				return fail(err)
			}
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fail(fmt.Errorf("Community-1 Vulkan block: nonfinite output[%d]", i))
		}
	}
	return output, s.stats.Output, ctx.Err()
}

// Close permanently blocks new calls and retries reverse teardown on failure.
// It never drains or restarts the device; unresolved work returns an error.
func (b *VulkanBasicBlock) Close() error {
	if b == nil || b.s == nil {
		return nil
	}
	s, err := b.acquire(context.Background())
	if err != nil {
		return err
	}
	defer func() { <-s.gate }()
	if s.closed {
		return nil
	}
	s.stopping = true
	var failures []error
	for i := len(s.resources) - 1; i >= 0; i-- {
		if resource := s.resources[i]; resource != nil {
			if err := resource.Close(); err != nil {
				failures = append(failures, err)
			} else {
				s.resources[i] = nil
			}
		}
	}
	if len(failures) != 0 {
		return errors.Join(failures...)
	}
	s.closed = true
	s.resources = nil
	s.plan = nil
	s.input = nil
	s.output = nil
	return nil
}
