package nemotrondiarization

import (
	"fmt"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// ptxLayerScratch is owned by one layer call or by a serialised tower. Its
// extents are capacities; launches still use the caller's exact row count.
// Index 8 is the wider fc1 activation; the other eight use projectedWidth.
type ptxLayerScratch struct {
	buffers [9]*ptx.Buffer
}

func newPTXLayerScratch(maxRows int) (scratch *ptxLayerScratch, err error) {
	if maxRows < 1 || maxRows > maxPreparedDiarizationRows {
		return nil, fmt.Errorf("invalid PTX layer scratch row capacity")
	}
	candidate := &ptxLayerScratch{}
	defer func() {
		if err != nil {
			candidate.close()
		}
	}()
	for i := range candidate.buffers {
		width := projectedWidth
		if i == 8 {
			width = diarizationIntermediate
		}
		candidate.buffers[i], err = ptx.Malloc(maxRows * width)
		if err != nil {
			return nil, err
		}
	}
	return candidate, nil
}

func (s *ptxLayerScratch) close() {
	if s == nil {
		return
	}
	for i := len(s.buffers) - 1; i >= 0; i-- {
		if s.buffers[i] != nil {
			s.buffers[i].Free()
			s.buffers[i] = nil
		}
	}
}

func (s *ptxLayerScratch) valid(rows int) bool {
	if s == nil || rows < 1 || rows > maxPreparedDiarizationRows {
		return false
	}
	for i, b := range s.buffers {
		width := projectedWidth
		if i == 8 {
			width = diarizationIntermediate
		}
		if b == nil || b.Ptr == 0 || b.Size < rows*width*4 {
			return false
		}
	}
	return true
}
