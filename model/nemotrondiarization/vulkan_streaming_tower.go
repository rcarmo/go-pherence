package nemotrondiarization

import (
	"context"
	"fmt"
	"sync"
)

// VulkanStreamingTower owns one maximum-row resident tower. Exact prefix
// views and checked plan rebinding preserve the live attention key count when
// the streaming cache grows. There is no silent CPU fallback. One instance
// belongs to one stream. Call Close, including after errors.
type VulkanStreamingTower struct {
	model  *OfflineAudioTower
	mu     sync.Mutex
	tower  *VulkanAudioTower
	rows   int
	closed bool
}

func NewVulkanStreamingTower(model *OfflineAudioTower) (*VulkanStreamingTower, error) {
	if model == nil || model.first == nil || len(model.remaining) != 30 {
		return nil, fmt.Errorf("invalid Nemotron Vulkan streaming tower model")
	}
	return &VulkanStreamingTower{model: model}, nil
}

func (s *VulkanStreamingTower) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.tower != nil {
		if err := s.tower.Close(); err != nil {
			return err // retain owner for retry after VulkanDrain
		}
		s.tower = nil
	}
	s.closed = true
	s.model = nil
	return nil
}

func (s *VulkanStreamingTower) Forward(ctx context.Context, input []float32, rows int) ([]float32, error) {
	if s == nil || ctx == nil {
		return nil, fmt.Errorf("invalid Nemotron Vulkan streaming tower")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.model == nil || rows < 1 || rows > maxPreparedDiarizationRows || len(input) != rows*projectedWidth {
		return nil, fmt.Errorf("invalid Nemotron Vulkan streaming tower input")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.tower == nil {
		candidate, err := NewVulkanAudioTower(ctx, s.model, maxPreparedDiarizationRows)
		if err != nil {
			if candidate != nil {
				s.tower, s.rows = candidate, maxPreparedDiarizationRows // constructor cleanup failed; preserve owner
			}
			return nil, err
		}
		s.tower, s.rows = candidate, maxPreparedDiarizationRows
	}
	return s.tower.ForwardRows(ctx, input, rows)
}

// EnableVulkanTower opts one stream into resident Vulkan encoder execution.
// No GPU work is done until a prepared window is ready. Errors never fall
// back silently to the CPU path.
func (s *PCMStreamingRequest) EnableVulkanTower() error {
	if s == nil || s.closed || s.window == nil || s.window.Tower == nil || s.window.VulkanTower != nil {
		return fmt.Errorf("invalid Nemotron Vulkan streaming request")
	}
	var err error
	s.window.VulkanTower, err = NewVulkanStreamingTower(s.window.Tower)
	return err
}

// CloseVulkanTower releases an optional streaming tower; it is safe to call
// after FinishContext and on errors. An unsuccessful close retains ownership.
func (s *PCMStreamingRequest) CloseVulkanTower() error {
	if s == nil || s.window == nil || s.window.VulkanTower == nil {
		return nil
	}
	if err := s.window.VulkanTower.Close(); err != nil {
		return fmt.Errorf("Nemotron streaming tower cleanup: %w", err)
	}
	s.window.VulkanTower = nil
	return nil
}
