package model

import (
	"fmt"
	"sync"

	"github.com/rcarmo/go-pherence/runtime/inferencesched"
)

var _ inferencesched.Work = (*Gemma4ScheduledWork)(nil)

// Gemma4ScheduledWork is a thin inferencesched adapter over a Gemma4 session.
// It owns no model state beyond serializing session access and buffering decode
// results for callers to drain after scheduler steps.
type Gemma4ScheduledWork struct {
	mu       sync.Mutex
	session  *Gemma4DecodeSession
	results  []DecodeResult
	canceled bool
	closed   bool
}

// NewGemma4ScheduledWork begins prefill for the full raw prompt exactly once
// and returns an inferencesched-compatible adapter around the session.
func NewGemma4ScheduledWork(session *Gemma4DecodeSession, prompt []int) (*Gemma4ScheduledWork, error) {
	if session == nil {
		return nil, fmt.Errorf("nil Gemma4 session")
	}
	if err := session.BeginPrefill(prompt); err != nil {
		return nil, err
	}
	return &Gemma4ScheduledWork{session: session}, nil
}

func (w *Gemma4ScheduledWork) RemainingPrefill() int {
	if w == nil {
		return 0
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.session.RemainingPrefill()
}

func (w *Gemma4ScheduledWork) Prefill(limit int) (int, error) {
	if w == nil {
		return 0, fmt.Errorf("nil Gemma4 scheduled work")
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	res, err := w.session.PrefillNext(limit)
	return res.ConsumedTokens, err
}

func (w *Gemma4ScheduledWork) Decode(limit int) (int, error) {
	if w == nil {
		return 0, fmt.Errorf("nil Gemma4 scheduled work")
	}
	if limit <= 0 {
		return 0, nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	consumed := 0
	for consumed < limit {
		finished, _ := w.session.Finished()
		if finished {
			break
		}
		res, err := w.session.DecodeStep()
		if err != nil {
			return consumed, err
		}
		w.results = append(w.results, cloneDecodeResult(res))
		consumed++
		if res.Finished {
			break
		}
	}
	return consumed, nil
}

// DrainResults returns and clears the buffered decode results.
func (w *Gemma4ScheduledWork) DrainResults() []DecodeResult {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.results) == 0 {
		return nil
	}
	out := make([]DecodeResult, len(w.results))
	for i, res := range w.results {
		out[i] = cloneDecodeResult(res)
	}
	clear(w.results)
	w.results = nil
	return out
}

func (w *Gemma4ScheduledWork) Finished() bool {
	if w == nil {
		return true
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	finished, _ := w.session.Finished()
	return finished
}

func (w *Gemma4ScheduledWork) Cancel() {
	if w == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.canceled = true
}

func (w *Gemma4ScheduledWork) Close() error {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.closed {
		return nil
	}
	w.closed = true
	return w.session.Close()
}

func cloneDecodeResult(res DecodeResult) DecodeResult {
	res.Logits = append([]float32(nil), res.Logits...)
	return res
}
