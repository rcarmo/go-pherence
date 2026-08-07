package nvidia

import (
	"fmt"
	"strings"
)

// capturedGraphSegment is a tiny shape-guarded wrapper around a captured CUDA
// graph. It is intentionally narrow: the live benchmark/test uses it to ensure
// replay only happens for the fixed batch-1 segment shape it was captured for.
type capturedGraphSegment struct {
	shapeKey string
	graph    *CapturedGraph
}

func newCapturedGraphSegment(shapeKey string, graph *CapturedGraph) (*capturedGraphSegment, error) {
	shapeKey = strings.TrimSpace(shapeKey)
	if shapeKey == "" {
		return nil, fmt.Errorf("empty CUDA graph shape key")
	}
	if graph == nil {
		return nil, fmt.Errorf("nil CUDA graph")
	}
	return &capturedGraphSegment{shapeKey: shapeKey, graph: graph}, nil
}

func (s *capturedGraphSegment) Launch(shapeKey string) error {
	if s == nil {
		return fmt.Errorf("nil CUDA graph segment")
	}
	if shapeKey != s.shapeKey {
		return fmt.Errorf("CUDA graph shape mismatch: got %q want %q", shapeKey, s.shapeKey)
	}
	if s.graph == nil {
		return fmt.Errorf("nil CUDA graph")
	}
	return s.graph.Launch()
}

func (s *capturedGraphSegment) Destroy() {
	if s == nil {
		return
	}
	if s.graph != nil {
		s.graph.Destroy()
		s.graph = nil
	}
}
