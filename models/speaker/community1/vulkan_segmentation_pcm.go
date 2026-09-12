package community1

import (
	"context"
	"errors"
	"fmt"
)

// VulkanSegmentationPCM combines the checked CPU lowered-filter SincNet frontend
// with a fixed-frame Vulkan recurrent/CPU-head owner. It is experimental and
// retains all existing strict SincNet qualification failures. No default uses it.
type VulkanSegmentationPCM struct {
	frontend *SincNet
	features *VulkanSegmentationFeatures
	samples  int
	grid     SincNetGrid
}

type vulkanSegmentationFactory func(context.Context, *SegmentationCheckpoint, int) (*VulkanSegmentationFeatures, error)

func NewVulkanSegmentationPCM(ctx context.Context, checkpoint *SegmentationCheckpoint, filters []float32, samples int) (*VulkanSegmentationPCM, error) {
	return newVulkanSegmentationPCM(ctx, checkpoint, filters, samples, NewVulkanSegmentationFeatures)
}

func newVulkanSegmentationPCM(ctx context.Context, checkpoint *SegmentationCheckpoint, filters []float32, samples int, makeFeatures vulkanSegmentationFactory) (*VulkanSegmentationPCM, error) {
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
	features, err := makeFeatures(ctx, checkpoint, grid.Frames)
	if err != nil {
		if features != nil {
			if closeErr := features.Close(); closeErr != nil {
				return &VulkanSegmentationPCM{frontend: frontend, features: features, samples: samples, grid: grid}, errors.Join(err, fmt.Errorf("Community-1 Vulkan PCM: rollback: %w", closeErr))
			}
		}
		return nil, err
	}
	if features == nil {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: feature constructor returned nil")
	}
	return &VulkanSegmentationPCM{frontend: frontend, features: features, samples: samples, grid: grid}, nil
}

func (m *VulkanSegmentationPCM) Stats() VulkanSegmentationStats {
	if m == nil || m.features == nil {
		return VulkanSegmentationStats{}
	}
	return m.features.Stats()
}

func (m *VulkanSegmentationPCM) Grid() SincNetGrid {
	if m == nil {
		return SincNetGrid{}
	}
	return m.grid
}

func (m *VulkanSegmentationPCM) ForwardPCM(ctx context.Context, pcm []float32, sincMode SincNetMode, headMode HeadMode) (*SegmentationPCMResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.frontend == nil || m.features == nil {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: uninitialized owner")
	}
	if len(pcm) != m.samples || (sincMode != SincNetScalarFMA && sincMode != SincNetSIMDFMA) || (headMode != HeadScalar && headMode != HeadSIMD) {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: invalid input/mode")
	}
	features, grid, err := m.frontend.Forward(ctx, pcm, sincMode)
	if err != nil {
		return nil, err
	}
	if grid != m.grid {
		return nil, fmt.Errorf("Community-1 Vulkan PCM: frontend grid changed")
	}
	scores, err := m.features.ForwardFeatures(ctx, features, grid.Frames, headMode)
	if err != nil {
		return nil, err
	}
	return &SegmentationPCMResult{Grid: grid, Classes: m.features.Stats().Classes, LogProbabilities: scores}, ctx.Err()
}

func (m *VulkanSegmentationPCM) Close() error {
	if m == nil || m.features == nil {
		return nil
	}
	if err := m.features.Close(); err != nil {
		return err
	}
	m.features = nil
	m.frontend = nil
	return nil
}
