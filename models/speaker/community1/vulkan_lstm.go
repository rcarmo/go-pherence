package community1

import (
	"context"
	"errors"
	"fmt"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// VulkanLSTM is an explicit fixed-frame resident owner for the complete
// unprojected Community-1 LSTM. It owns copied weights, mutable state/scratch,
// one direction kernel and one plan per layer. Calls are serialized.
type VulkanLSTM struct{ s *vulkanLSTMState }

type VulkanLSTMStats struct {
	Frames, InputSize, HiddenSize, Layers, Directions, Plans, Stages int
	WeightBytes, ScratchBytes                                        uint64
}

type vulkanLSTMCloser interface{ Close() error }

type vulkanLSTMState struct {
	gate             chan struct{}
	stopping, closed bool
	stats            VulkanLSTMStats
	resources        []vulkanLSTMCloser
	plans            []*vk.VkF32Plan
	input, output    *vk.VkTensorF32
	hidden, cell     []*vk.VkTensorF32
	op               *vk.VkLSTMSequenceF32
}

func NewVulkanLSTM(ctx context.Context, source *LSTM, frames int) (*VulkanLSTM, error) {
	return newVulkanLSTM(ctx, source, frames, vk.NewVkF32Plan)
}

func newVulkanLSTM(ctx context.Context, source *LSTM, frames int, makePlan func(context.Context, []vk.VkF32Stage) (*vk.VkF32Plan, error)) (result *VulkanLSTM, err error) {
	if makePlan == nil {
		return nil, fmt.Errorf("Community-1 Vulkan LSTM: nil plan constructor")
	}
	layout, err := describeVulkanLSTM(ctx, source, frames)
	if err != nil {
		return nil, err
	}
	limits, err := vk.VulkanLimits()
	if err != nil {
		return nil, err
	}
	all := append(append([]vulkanBlockTensor(nil), layout.weights...), layout.scratch...)
	arenaBytes, err := vulkanBlockArenaBytes(all, limits.StorageBufferOffsetAlignment)
	if err != nil {
		return nil, err
	}
	if arenaBytes > uint64(limits.StorageBufferRange) || arenaBytes > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("Community-1 Vulkan LSTM: arena exceeds device range")
	}
	s := &vulkanLSTMState{gate: make(chan struct{}, 1), stats: VulkanLSTMStats{Frames: frames, InputSize: layout.cfg.InputSize, HiddenSize: layout.cfg.HiddenSize, Layers: layout.cfg.NumLayers, Directions: layout.directions, Plans: len(layout.plans), Stages: layout.cfg.NumLayers * layout.directions, WeightBytes: layout.weightBytes, ScratchBytes: layout.scratchBytes}}
	owner := &VulkanLSTM{s: s}
	defer func() {
		if err != nil {
			if closeErr := owner.Close(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Community-1 Vulkan LSTM: rollback: %w", closeErr))
			}
		}
	}()
	if s.op, err = vk.NewVkLSTMSequenceF32(ctx); err != nil {
		return nil, err
	}
	s.resources = append(s.resources, s.op)
	arena, err := vk.NewVkTensorArena(ctx, int(arenaBytes))
	if err != nil {
		return nil, err
	}
	s.resources = append(s.resources, arena)
	tensors := make(map[string]*vk.VkTensorF32, len(all))
	for _, spec := range all {
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
	s.input, s.output = tensors[layout.inputName], tensors[layout.outputName]
	for i := range layout.hiddenNames {
		s.hidden = append(s.hidden, tensors[layout.hiddenNames[i]])
		s.cell = append(s.cell, tensors[layout.cellNames[i]])
	}
	for _, steps := range layout.plans {
		stages := make([]vk.VkF32Stage, 0, len(steps))
		for _, step := range steps {
			stage, stageErr := s.op.Stage(ctx, tensors[step.output], tensors[step.input], tensors[step.weightIH], tensors[step.weightHH], tensors[step.biasIH], tensors[step.biasHH], tensors[step.hidden], tensors[step.cell], step.outputOffset, step.reverse)
			if stageErr != nil {
				return nil, stageErr
			}
			stages = append(stages, stage)
		}
		plan, planErr := makePlan(ctx, stages)
		if plan != nil {
			s.resources = append(s.resources, plan)
			s.plans = append(s.plans, plan)
		}
		if planErr != nil {
			return nil, planErr
		}
		if plan == nil {
			return nil, fmt.Errorf("Community-1 Vulkan LSTM: plan constructor returned nil")
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	return owner, nil
}

func (l *VulkanLSTM) acquire(ctx context.Context) (*vulkanLSTMState, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan LSTM: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if l == nil || l.s == nil {
		return nil, fmt.Errorf("Community-1 Vulkan LSTM: uninitialized owner")
	}
	s := l.s
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

func (l *VulkanLSTM) Stats() VulkanLSTMStats {
	if l == nil || l.s == nil {
		return VulkanLSTMStats{}
	}
	return l.s.stats
}

func (l *VulkanLSTM) Forward(ctx context.Context, input, hidden, cell []float32) (*LSTMResult, error) {
	s, err := l.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { <-s.gate }()
	if s.stopping || s.closed {
		return nil, vk.ErrVulkanClosed
	}
	stateSize := s.stats.Layers * s.stats.Directions * s.stats.HiddenSize
	if len(input) != s.stats.Frames*s.stats.InputSize || !(hidden == nil && cell == nil) && (len(hidden) != stateSize || len(cell) != stateSize) {
		return nil, fmt.Errorf("Community-1 Vulkan LSTM: invalid input/state")
	}
	for _, values := range [][]float32{input, hidden, cell} {
		if err := finiteLSTM(ctx, values); err != nil {
			return nil, err
		}
	}
	if err := s.input.Upload(ctx, input); err != nil {
		return nil, err
	}
	zero := make([]float32, s.stats.HiddenSize)
	for i := range s.hidden {
		h := zero
		c := zero
		if hidden != nil {
			offset := i * s.stats.HiddenSize
			h = hidden[offset : offset+s.stats.HiddenSize]
			c = cell[offset : offset+s.stats.HiddenSize]
		}
		if err := s.hidden[i].Upload(ctx, h); err != nil {
			return nil, err
		}
		if err := s.cell[i].Upload(ctx, c); err != nil {
			return nil, err
		}
	}
	for i, plan := range s.plans {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := plan.Run(ctx); err != nil {
			return nil, fmt.Errorf("Community-1 Vulkan LSTM: plan%d: %w", i, err)
		}
	}
	result := &LSTMResult{Output: make([]float32, s.output.Elements()), Hidden: make([]float32, stateSize), Cell: make([]float32, stateSize)}
	if err := s.output.Download(ctx, result.Output); err != nil {
		return nil, err
	}
	for i := range s.hidden {
		offset := i * s.stats.HiddenSize
		if err := s.hidden[i].Download(ctx, result.Hidden[offset:offset+s.stats.HiddenSize]); err != nil {
			return nil, err
		}
		if err := s.cell[i].Download(ctx, result.Cell[offset:offset+s.stats.HiddenSize]); err != nil {
			return nil, err
		}
	}
	for _, values := range [][]float32{result.Output, result.Hidden, result.Cell} {
		if err := finiteLSTM(ctx, values); err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return result, nil
}

func (l *VulkanLSTM) Close() error {
	if l == nil || l.s == nil {
		return nil
	}
	s, err := l.acquire(context.Background())
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
	s.hidden = nil
	s.cell = nil
	return nil
}
