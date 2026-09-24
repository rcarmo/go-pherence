package pockettts

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// flowTapeScratch owns bounded per-frame tape slices. Reset invalidates every
// slice returned by take; callers must finish both forward and backward first.
// Standalone public APIs pass nil and retain their owned-result behaviour.
type flowTapeScratch struct {
	buffers   [][]float32
	used      int
	fallbacks int
}

func newFlowTapeScratch(flow *FlowHeadCPU) (*flowTapeScratch, error) {
	if flow == nil {
		return nil, fmt.Errorf("nil Pocket TTS flow tape scratch model")
	}
	blocks, ok := checked.MulInt(40, len(flow.Blocks))
	if !ok {
		return nil, fmt.Errorf("Pocket TTS flow tape scratch slots overflow")
	}
	times, ok := checked.MulInt(24, len(flow.Time))
	if !ok {
		return nil, fmt.Errorf("Pocket TTS flow tape scratch slots overflow")
	}
	slots, ok := checked.AddInt(64, blocks)
	if ok {
		slots, ok = checked.AddInt(slots, times)
	}
	if !ok || slots <= 0 {
		return nil, fmt.Errorf("Pocket TTS flow tape scratch slots overflow")
	}
	return &flowTapeScratch{buffers: make([][]float32, slots)}, nil
}

func (s *flowTapeScratch) reset() {
	if s != nil {
		s.used = 0
	}
}

func (s *flowTapeScratch) take(n int) []float32 {
	if s == nil {
		return make([]float32, n)
	}
	if s.used >= len(s.buffers) {
		s.fallbacks++
		return make([]float32, n)
	}
	index := s.used
	s.used++
	if cap(s.buffers[index]) < n {
		s.buffers[index] = make([]float32, n)
	}
	values := s.buffers[index][:n]
	clear(values)
	return values
}

func (s *flowTapeScratch) copy(values []float32) []float32 {
	out := s.take(len(values))
	copy(out, values)
	return out
}
