package community1

import (
	"context"
	"errors"
	"fmt"
)

// VulkanSegmentationPCM combines the checked CPU lowered-filter SincNet frontend
// with a fixed-frame Vulkan recurrent/CPU-head owner. Copies share one serialized
// lifecycle. It is experimental and retains all existing strict SincNet
// qualification failures. No default uses it.
type VulkanSegmentationPCM struct{ s *vulkanSegmentationPCMState }

type vulkanSegmentationPCMState struct {
	gate             chan struct{}
	stopping, closed bool
	frontend         *SincNet
	features         *VulkanSegmentationFeatures
	samples          int
	grid             SincNetGrid
	stats            VulkanSegmentationStats
}

type vulkanSegmentationFactory func(context.Context, *SegmentationCheckpoint, int) (*VulkanSegmentationFeatures, error)

func NewVulkanSegmentationPCM(ctx context.Context, checkpoint *SegmentationCheckpoint, filters []float32, samples int) (*VulkanSegmentationPCM, error) {
	return newVulkanSegmentationPCM(ctx, checkpoint, filters, samples, NewVulkanSegmentationFeatures)
}

func newVulkanSegmentationPCM(ctx context.Context, checkpoint *SegmentationCheckpoint, filters []float32, samples int, makeFeatures vulkanSegmentationFactory) (result *VulkanSegmentationPCM, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if checkpoint == nil || checkpoint.recurrent == nil || checkpoint.head == nil {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: invalid checkpoint")
	}
	if makeFeatures == nil {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: nil feature constructor")
	}
	grid, err := checkpoint.Grid(samples)
	if err != nil {
		return nil, err
	}
	frontend, err := NewSincNetWithFilters(ctx, checkpoint.cfg.SincNetStride, checkpoint.sincnet, filters)
	if err != nil {
		return nil, err
	}
	state := &vulkanSegmentationPCMState{gate: make(chan struct{}, 1), frontend: frontend, samples: samples, grid: grid}
	owner := &VulkanSegmentationPCM{s: state}
	defer func() {
		if err != nil {
			if closeErr := owner.Close(); closeErr != nil {
				result = owner
				err = errors.Join(err, fmt.Errorf("Community-1 Vulkan PCM: rollback: %w", closeErr))
			}
		}
	}()
	// Adopt a partial owner before checking the constructor error. Feature
	// construction can retain native resources when its own rollback must retry.
	state.features, err = makeFeatures(ctx, checkpoint, grid.Frames)
	if err != nil {
		return nil, err
	}
	if state.features == nil {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: feature constructor returned nil")
	}
	state.stats = state.features.Stats()
	return owner, nil
}

func (m *VulkanSegmentationPCM) acquire(ctx context.Context) (*vulkanSegmentationPCMState, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.s == nil {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: uninitialized owner")
	}
	state := m.s
	select {
	case state.gate <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-state.gate
			return nil, err
		}
		return state, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (m *VulkanSegmentationPCM) Stats() VulkanSegmentationStats {
	if m == nil || m.s == nil {
		return VulkanSegmentationStats{}
	}
	return m.s.stats
}

func (m *VulkanSegmentationPCM) Grid() SincNetGrid {
	if m == nil || m.s == nil {
		return SincNetGrid{}
	}
	return m.s.grid
}

func (m *VulkanSegmentationPCM) ForwardPCM(ctx context.Context, pcm []float32, sincMode SincNetMode, headMode HeadMode) (*SegmentationPCMResult, error) {
	state, err := m.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { <-state.gate }()
	if state.stopping || state.closed || state.frontend == nil || state.features == nil {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: closed/uninitialized owner")
	}
	if len(pcm) != state.samples || (sincMode != SincNetScalarFMA && sincMode != SincNetSIMDFMA) || (headMode != HeadScalar && headMode != HeadSIMD) {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: invalid input/mode")
	}
	features, grid, err := state.frontend.Forward(ctx, pcm, sincMode)
	if err != nil {
		return nil, err
	}
	if grid != state.grid {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: frontend grid changed")
	}
	scores, err := state.features.ForwardFeatures(ctx, features, grid.Frames, headMode)
	if err != nil {
		return nil, err
	}
	return &SegmentationPCMResult{Grid: grid, Classes: state.stats.Classes, LogProbabilities: scores}, ctx.Err()
}

func (m *VulkanSegmentationPCM) Close() error {
	if m == nil || m.s == nil {
		return nil
	}
	state, err := m.acquire(context.Background())
	if err != nil {
		return err
	}
	defer func() { <-state.gate }()
	if state.closed {
		return nil
	}
	state.stopping = true
	if state.features != nil {
		if err := state.features.Close(); err != nil {
			return err
		}
	}
	state.closed = true
	state.features = nil
	state.frontend = nil
	return nil
}
