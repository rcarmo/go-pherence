package community1

import (
	"context"
	"errors"
	"fmt"
	"math"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// VulkanResNetTrunk is an explicit fixed-frame resident owner for the complete
// Community-1 WeSpeaker stem and ResNet34 trunk. Pooling and embedding projection
// remain host-owned. Copies share ownership/serialisation. No default selects it.
type VulkanResNetTrunk struct{ s *vulkanResNetState }

type VulkanResNetStats struct {
	Frames, Plans, Stages     int
	Input, Output             CHWShape
	WeightBytes, ScratchBytes uint64
}

type vulkanResNetCloser interface{ Close() error }

type vulkanResNetState struct {
	gate             chan struct{}
	stopping, closed bool
	stats            VulkanResNetStats
	resources        []vulkanResNetCloser
	plans            []*vk.VkF32Plan
	input, output    *vk.VkTensorF32
	conv             *vk.VkConv2DCHWF32
	affine           *vk.VkChannelAffineReLUF32
	add              *vk.VkAddF32
}

func NewVulkanResNetTrunk(ctx context.Context, source *WeSpeakerResNet34, frames int) (*VulkanResNetTrunk, error) {
	return newVulkanResNetTrunk(ctx, source, frames, vk.NewVkF32Plan)
}

func newVulkanResNetTrunk(ctx context.Context, source *WeSpeakerResNet34, frames int, makePlan func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error)) (result *VulkanResNetTrunk, err error) {
	if makePlan == nil {
		return nil, fmt.Errorf("Community-1 Vulkan trunk: nil plan constructor")
	}
	layout, err := describeVulkanResNetTrunk(ctx, source, frames)
	if err != nil {
		return nil, err
	}
	limits, err := vk.VulkanLimits()
	if err != nil {
		return nil, err
	}
	all := make([]vulkanBlockTensor, 0, len(layout.scratch)+128)
	for _, group := range layout.weights {
		all = append(all, group...)
	}
	all = append(all, layout.scratch...)
	arenaBytes, err := vulkanBlockArenaBytes(all, limits.StorageBufferOffsetAlignment)
	if err != nil {
		return nil, err
	}
	if arenaBytes > uint64(limits.StorageBufferRange) || arenaBytes > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("Community-1 Vulkan trunk: arena exceeds device range")
	}
	stages := 0
	for _, plan := range layout.plans {
		stages += len(plan)
	}
	s := &vulkanResNetState{gate: make(chan struct{}, 1), stats: VulkanResNetStats{Frames: frames, Plans: len(layout.plans), Stages: stages, Input: layout.input, Output: layout.output, WeightBytes: layout.weightBytes, ScratchBytes: layout.scratchBytes}}
	owner := &VulkanResNetTrunk{s: s}
	defer func() {
		if err != nil {
			if closeErr := owner.Close(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Community-1 Vulkan trunk: rollback: %w", closeErr))
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
	tensors := make(map[string]*vk.VkTensorF32, len(all))
	for _, spec := range all {
		if tensors[spec.name] != nil {
			return nil, fmt.Errorf("Community-1 Vulkan trunk: duplicate tensor %q", spec.name)
		}
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
	s.input, s.output = tensors["input"], tensors[layout.outputName]
	for _, steps := range layout.plans {
		planStages := make([]vk.VkF32Stage, 0, len(steps))
		for _, step := range steps {
			var stage vk.VkF32Stage
			switch step.op {
			case "conv":
				stage, err = s.conv.Stage(ctx, tensors[step.out], tensors[step.x], tensors[step.a], step.kernel, step.stride, step.padding)
			case "affine":
				stage, err = s.affine.Stage(ctx, tensors[step.out], tensors[step.x], tensors[step.a], tensors[step.b], step.relu)
			case "add":
				stage, err = s.add.Stage(ctx, tensors[step.out], tensors[step.x], tensors[step.a])
			default:
				return nil, fmt.Errorf("Community-1 Vulkan trunk: unknown graph operator %q", step.op)
			}
			if err != nil {
				return nil, err
			}
			planStages = append(planStages, stage)
		}
		plan, planErr := makePlan(ctx, planStages)
		if plan != nil {
			s.resources = append(s.resources, plan)
			s.plans = append(s.plans, plan)
		}
		if planErr != nil {
			return nil, planErr
		}
		if plan == nil {
			return nil, fmt.Errorf("Community-1 Vulkan trunk: plan constructor returned nil")
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return owner, nil
}

func (t *VulkanResNetTrunk) acquire(ctx context.Context) (*vulkanResNetState, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan trunk: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if t == nil || t.s == nil {
		return nil, fmt.Errorf("Community-1 Vulkan trunk: uninitialized owner")
	}
	s := t.s
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

func (t *VulkanResNetTrunk) Stats() VulkanResNetStats {
	if t == nil || t.s == nil {
		return VulkanResNetStats{}
	}
	return t.s.stats
}

// ForwardFrames accepts frame-major [frames,melBins], performs the same explicit
// transpose as the CPU trunk, then runs 17 resident plans and downloads final CHW.
func (t *VulkanResNetTrunk) ForwardFrames(ctx context.Context, fbank []float32, frames int) ([]float32, CHWShape, error) {
	fail := func(err error) ([]float32, CHWShape, error) { return nil, CHWShape{}, err }
	s, err := t.acquire(ctx)
	if err != nil {
		return fail(err)
	}
	defer func() { <-s.gate }()
	if s.stopping || s.closed {
		return fail(vk.ErrVulkanClosed)
	}
	if frames != s.stats.Frames || len(fbank) != frames*s.stats.Input.Frequency {
		return fail(fmt.Errorf("Community-1 Vulkan trunk: Fbank geometry mismatch"))
	}
	input := make([]float32, len(fbank))
	for frame := 0; frame < frames; frame++ {
		if frame%256 == 0 {
			if err := ctx.Err(); err != nil {
				return fail(err)
			}
		}
		for frequency := 0; frequency < s.stats.Input.Frequency; frequency++ {
			value := fbank[frame*s.stats.Input.Frequency+frequency]
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fail(fmt.Errorf("Community-1 Vulkan trunk: nonfinite Fbank[%d]", frame*s.stats.Input.Frequency+frequency))
			}
			input[frequency*frames+frame] = value
		}
	}
	if err := s.input.Upload(ctx, input); err != nil {
		return fail(err)
	}
	for i, plan := range s.plans {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		if err := plan.Run(ctx); err != nil {
			return fail(fmt.Errorf("Community-1 Vulkan trunk: plan%d: %w", i, err))
		}
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
			return fail(fmt.Errorf("Community-1 Vulkan trunk: nonfinite output[%d]", i))
		}
	}
	return output, s.stats.Output, ctx.Err()
}

func (t *VulkanResNetTrunk) Close() error {
	if t == nil || t.s == nil {
		return nil
	}
	s, err := t.acquire(context.Background())
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
	s.plans = nil
	s.input = nil
	s.output = nil
	return nil
}
