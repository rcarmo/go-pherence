package community1

import (
	"context"
	"errors"
	"fmt"

	vk "github.com/rcarmo/go-pherence/backends/vulkan"
)

// VulkanSegmentationFeatures is an explicit hybrid fixed-frame owner: recurrent
// inference is resident Vulkan and the copied powerset head remains on CPU. It
// consumes SincNet features, not PCM. No model/service default selects it.
type VulkanSegmentationFeatures struct{ s *vulkanSegmentationState }

type VulkanSegmentationStats struct {
	Frames, InputSize, RecurrentWidth, Classes int
	LSTM                                       VulkanLSTMStats
	HeadBytes                                  uint64
}

type vulkanSegmentationState struct {
	gate             chan struct{}
	stopping, closed bool
	stats            VulkanSegmentationStats
	recurrent        *VulkanLSTM
	head             *SegmentationHead
}

type vulkanLSTMFactory func(context.Context, *LSTM, int) (*VulkanLSTM, error)

func NewVulkanSegmentationFeatures(ctx context.Context, source *SegmentationCheckpoint, frames int) (*VulkanSegmentationFeatures, error) {
	return newVulkanSegmentationFeatures(ctx, source, frames, NewVulkanLSTM)
}

func newVulkanSegmentationFeatures(ctx context.Context, source *SegmentationCheckpoint, frames int, makeLSTM vulkanLSTMFactory) (result *VulkanSegmentationFeatures, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan segmentation: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil || source.recurrent == nil || source.head == nil {
		return nil, fmt.Errorf("Community-1 Vulkan segmentation: invalid source")
	}
	if makeLSTM == nil {
		return nil, fmt.Errorf("Community-1 Vulkan segmentation: nil recurrent constructor")
	}
	if frames < 1 || frames > MaxPowersetFrames || source.cfg.LSTM.InputSize != 60 || source.cfg.Head.InputSize != source.cfg.LSTM.HiddenSize*lstmDirections(source.cfg.LSTM) {
		return nil, fmt.Errorf("Community-1 Vulkan segmentation: invalid geometry")
	}
	if err := checkLSTMConfig(source.cfg.LSTM); err != nil {
		return nil, err
	}
	classes, err := checkHeadConfig(source.cfg.Head)
	if err != nil || classes != source.head.classes {
		return nil, fmt.Errorf("Community-1 Vulkan segmentation: invalid head geometry")
	}
	head, err := NewSegmentationHead(ctx, source.head.cfg, source.head.layers, source.head.classifier)
	if err != nil {
		return nil, err
	}
	headBytes := uint64(0)
	for _, layer := range append(append([]HeadLinear(nil), head.layers...), head.classifier) {
		headBytes += uint64(len(layer.Weight)+len(layer.Bias)) * 4
	}
	recurrent, err := makeLSTM(ctx, source.recurrent, frames)
	state := &vulkanSegmentationState{gate: make(chan struct{}, 1), recurrent: recurrent, head: head}
	owner := &VulkanSegmentationFeatures{s: state}
	if err != nil {
		if recurrent != nil {
			if closeErr := recurrent.Close(); closeErr != nil {
				state.stopping = true
				return owner, errors.Join(err, fmt.Errorf("Community-1 Vulkan segmentation: rollback: %w", closeErr))
			}
		}
		return nil, err
	}
	if recurrent == nil {
		return nil, fmt.Errorf("Community-1 Vulkan segmentation: recurrent constructor returned nil")
	}
	state.stats = VulkanSegmentationStats{Frames: frames, InputSize: source.cfg.LSTM.InputSize, RecurrentWidth: source.cfg.Head.InputSize, Classes: classes, LSTM: recurrent.Stats(), HeadBytes: headBytes}
	return owner, nil
}

func (s *VulkanSegmentationFeatures) acquire(ctx context.Context) (*vulkanSegmentationState, error) {
	if ctx == nil {
		return nil, fmt.Errorf("Community-1 Vulkan segmentation: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s == nil || s.s == nil {
		return nil, fmt.Errorf("Community-1 Vulkan segmentation: uninitialized owner")
	}
	state := s.s
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

func (s *VulkanSegmentationFeatures) Stats() VulkanSegmentationStats {
	if s == nil || s.s == nil {
		return VulkanSegmentationStats{}
	}
	return s.s.stats
}

func (s *VulkanSegmentationFeatures) ForwardFeatures(ctx context.Context, input []float32, frames int, headMode HeadMode) ([]float32, error) {
	state, err := s.acquire(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { <-state.gate }()
	if state.stopping || state.closed {
		return nil, vk.ErrVulkanClosed
	}
	if frames != state.stats.Frames || len(input) != frames*state.stats.InputSize || (headMode != HeadScalar && headMode != HeadSIMD) {
		return nil, fmt.Errorf("Community-1 Vulkan segmentation: invalid input/mode")
	}
	if err := finiteLSTM(ctx, input); err != nil {
		return nil, err
	}
	sequence, err := state.recurrent.Forward(ctx, input, nil, nil)
	if err != nil {
		return nil, err
	}
	return state.head.Forward(ctx, sequence.Output, frames, headMode)
}

func (s *VulkanSegmentationFeatures) Close() error {
	if s == nil || s.s == nil {
		return nil
	}
	state, err := s.acquire(context.Background())
	if err != nil {
		return err
	}
	defer func() { <-state.gate }()
	if state.closed {
		return nil
	}
	state.stopping = true
	if state.recurrent != nil {
		if err := state.recurrent.Close(); err != nil {
			return err
		}
	}
	state.closed = true
	state.recurrent = nil
	state.head = nil
	return nil
}
