package nemotrondiarization

import (
	"errors"
	"fmt"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// ptxLayerScratch is owned by one layer call or by a serialised tower. Its
// extents are capacities; launches still use the caller's exact row count.
// Index 8 is the wider fc1 activation; the other eight use projectedWidth.
type ptxLayerScratch struct {
	buffers  [9]*ptx.Buffer
	terminal error
}

func newPTXLayerScratch(maxRows int) (scratch *ptxLayerScratch, err error) {
	if maxRows < 1 || maxRows > maxPreparedDiarizationRows {
		return nil, fmt.Errorf("invalid PTX layer scratch row capacity")
	}
	candidate := &ptxLayerScratch{}
	defer func() {
		if err != nil {
			if syncErr := ptxCleanupSync(); syncErr != nil {
				candidate.terminal = fmt.Errorf("PTX scratch construction sync: %w", syncErr)
				err = errors.Join(err, candidate.terminal)
				scratch = candidate // completion uncertain: do not free
				return
			}
			if closeErr := candidate.close(); closeErr != nil {
				err = errors.Join(err, closeErr)
				scratch = candidate // retain unconfirmed owners
			}
		}
	}()
	for i := range candidate.buffers {
		width := projectedWidth
		if i == 8 {
			width = diarizationIntermediate
		}
		candidate.buffers[i], err = ptxCleanupMalloc(maxRows * width)
		if err != nil {
			return nil, err
		}
	}
	return candidate, nil
}

func (s *ptxLayerScratch) close() error { return s.closeChecked(ptxCleanupFree) }

// Keep failed owners reachable; do not retry after a driver error without
// separately establishing context health. A successful free is never repeated.
func (s *ptxLayerScratch) closeChecked(free func(*ptx.Buffer) error) error {
	if s == nil {
		return nil
	}
	if s.terminal != nil {
		return s.terminal
	}
	var err error
	for i := len(s.buffers) - 1; i >= 0; i-- {
		if s.buffers[i] != nil {
			if e := free(s.buffers[i]); e != nil {
				err = errors.Join(err, e)
				s.terminal = err
				break
			}
			s.buffers[i] = nil
		}
	}
	return err
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
