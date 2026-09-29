package nemotrondiarization

import (
	"context"
	"fmt"
	"sync"
)

// PTXStreamingTower owns one maximum-row resident tower per stream. Exact
// attention key counts use the current row count, never unmasked padding.
// There is no silent CPU fallback. Call Close, including after an error.
type PTXStreamingTower struct {
	model  *OfflineAudioTower
	mu     sync.Mutex
	tower  *PTXAudioTower
	closed bool
}

func NewPTXStreamingTower(model *OfflineAudioTower) (*PTXStreamingTower, error) {
	if model == nil || model.first == nil || len(model.remaining) != 30 {
		return nil, fmt.Errorf("invalid PTX streaming tower model")
	}
	return &PTXStreamingTower{model: model}, nil
}

func (s *PTXStreamingTower) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	if s.tower != nil {
		s.tower.Close()
		s.tower = nil
	}
	s.model = nil
}

func (s *PTXStreamingTower) Forward(ctx context.Context, input []float32, rows int) ([]float32, error) {
	if s == nil || ctx == nil {
		return nil, fmt.Errorf("invalid PTX streaming tower")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed || s.model == nil || rows < 1 || rows > maxPreparedDiarizationRows || len(input) != rows*projectedWidth {
		return nil, fmt.Errorf("invalid PTX streaming tower input")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s.tower == nil {
		candidate, err := NewPTXAudioTower(s.model, maxPreparedDiarizationRows)
		if err != nil {
			return nil, err
		}
		s.tower = candidate
	}
	return s.tower.ForwardRows(ctx, input, rows)
}

// EnablePTXTower opts a stream into PTX resident encoder execution. It leaves
// frontend, first layer, head and speaker cache on CPU.
func (s *PCMStreamingRequest) EnablePTXTower() error {
	if s == nil || s.closed || s.window == nil || s.window.Tower == nil || s.window.PTXTower != nil || s.window.VulkanTower != nil {
		return fmt.Errorf("invalid PTX streaming request")
	}
	var err error
	s.window.PTXTower, err = NewPTXStreamingTower(s.window.Tower)
	return err
}

func (s *PCMStreamingRequest) ClosePTXTower() error {
	if s == nil || s.window == nil || s.window.PTXTower == nil {
		return nil
	}
	s.window.PTXTower.Close()
	s.window.PTXTower = nil
	return nil
}
