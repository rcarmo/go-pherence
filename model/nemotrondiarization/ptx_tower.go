package nemotrondiarization

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"

	ptx "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// PTXAudioTower keeps the remaining 30 unmasked layers and final norm weights
// on CUDA. Layer 0 and the request frontend/head remain on CPU. ForwardRows
// uses its exact key count and position origin; the maximum row capacity only
// bounds owned weights and the prepared frequency table. Call Close.
type PTXAudioTower struct {
	mu           sync.Mutex
	closed       bool
	terminal     error         // failed sync/free: no further CUDA calls from this owner
	pending      []*ptx.Buffer // transient owners retained after uncertain completion
	first        *Layer0Complete
	layers       []*PTXAudioLayer
	weight, bias *ptx.Buffer
	scratch      *ptxLayerScratch
	maxRows      int
}

func NewPTXAudioTower(source *OfflineAudioTower, maxRows int) (result *PTXAudioTower, err error) {
	if source == nil || source.first == nil || len(source.remaining) != 30 ||
		len(source.finalWeight) != projectedWidth || len(source.finalBias) != projectedWidth ||
		maxRows < 1 || maxRows > maxPreparedDiarizationRows || !ptx.SgemmReady() {
		return nil, fmt.Errorf("invalid PTX diarization tower model or row capacity")
	}
	for _, values := range [][]float32{source.finalWeight, source.finalBias} {
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, fmt.Errorf("non-finite PTX diarization tower final norm")
			}
		}
	}
	t := &PTXAudioTower{first: source.first, maxRows: maxRows, layers: make([]*PTXAudioLayer, 0, 30)}
	defer func() {
		if err != nil {
			closeErr := t.Close()
			err = errors.Join(err, closeErr)
			if closeErr != nil {
				result = t // retain failed owners for process-level recovery
			}
		}
	}()
	for index, model := range source.remaining {
		var layer *PTXAudioLayer
		layer, err = NewPTXAudioLayer(model, maxRows)
		if err != nil {
			if layer != nil {
				t.layers = append(t.layers, layer) // failed constructor retains owners
				t.terminal = fmt.Errorf("PTX tower layer construction: %w", err)
			}
			return nil, fmt.Errorf("PTX tower layer %d: %w", index+1, err)
		}
		t.layers = append(t.layers, layer)
	}
	upload := func(data []float32) (*ptx.Buffer, error) {
		b, e := ptxCleanupMalloc(len(data))
		if e != nil {
			return nil, e
		}
		if e = b.Upload(data); e != nil {
			// Preserve ownership until the tower's checked, synchronized Close.
			return b, e
		}
		return b, nil
	}
	t.weight, err = upload(source.finalWeight)
	if err != nil {
		return nil, err
	}
	t.bias, err = upload(source.finalBias)
	if err != nil {
		return nil, err
	}
	t.scratch, err = newPTXLayerScratch(maxRows)
	if err != nil {
		if t.scratch != nil {
			t.terminal = fmt.Errorf("PTX tower scratch construction: %w", err)
		}
		return nil, err
	}
	return t, nil
}

// Close serializes with ForwardRows. A sync/free failure seals this owner:
// retain every unconfirmed allocation and reject retries until the process
// owner performs separate recovery. A repeated Close returns the same error.
func (t *PTXAudioTower) Close() error {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.terminal != nil {
		return t.terminal
	}
	if t.closed {
		return nil
	}
	if err := ptxCleanupSync(); err != nil {
		t.terminal = fmt.Errorf("PTX tower close: %w", err)
		return t.terminal
	}
	if t.scratch != nil {
		if err := t.scratch.close(); err != nil {
			t.terminal = fmt.Errorf("PTX tower scratch close: %w", err)
			return t.terminal
		}
		t.scratch = nil
	}
	for _, owner := range []struct {
		name string
		buf  **ptx.Buffer
	}{{"bias", &t.bias}, {"weight", &t.weight}} {
		if *owner.buf != nil {
			if err := ptxCleanupFree(*owner.buf); err != nil {
				t.terminal = fmt.Errorf("PTX tower %s close: %w", owner.name, err)
				return t.terminal
			}
			*owner.buf = nil
		}
	}
	for i := len(t.layers) - 1; i >= 0; i-- {
		if t.layers[i] != nil {
			if err := t.layers[i].Close(); err != nil {
				t.terminal = fmt.Errorf("PTX tower layer %d close: %w", i+1, err)
				return t.terminal
			}
			t.layers[i] = nil
		}
	}
	t.closed = true
	t.layers, t.first, t.weight, t.bias, t.scratch = nil, nil, nil, nil, nil
	return nil
}

// releaseForwardOwners is called under t.mu after work may have been queued.
// Never retry a failed free in-process; successful frees have zeroed pointers.
func (t *PTXAudioTower) releaseForwardOwners(owners []*ptx.Buffer) error {
	if t.terminal != nil {
		t.pending = append(t.pending, owners...)
		return t.terminal // prior failed sync: no second sync/free can erase uncertainty
	}
	if syncErr := ptxCleanupSync(); syncErr != nil {
		t.terminal = fmt.Errorf("PTX tower forward sync: %w", syncErr)
		t.pending = append(t.pending, owners...)
		return t.terminal
	}
	for i := len(owners) - 1; i >= 0; i-- {
		if freeErr := ptxCleanupFree(owners[i]); freeErr != nil {
			t.terminal = fmt.Errorf("PTX tower forward free: %w", freeErr)
			t.pending = append(t.pending, owners...)
			return t.terminal
		}
	}
	return nil
}

// ForwardRows returns an owned final-normalised output for one exact unmasked
// stacking prefix. Context is checked between bounded layer launches. Failure
// returns no partial output and does not silently switch backends.
func (t *PTXAudioTower) ForwardRows(ctx context.Context, stacked []float32, rows int) (result []float32, err error) {
	if t == nil || ctx == nil {
		return nil, fmt.Errorf("invalid PTX diarization tower or context")
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.terminal != nil {
		return nil, t.terminal
	}
	if t.closed || t.first == nil || len(t.layers) != 30 || rows < 1 || rows > t.maxRows || len(stacked) != rows*projectedWidth || !t.scratch.valid(rows) {
		return nil, fmt.Errorf("invalid PTX diarization tower window")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, v := range stacked {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite PTX diarization tower input")
		}
	}
	first, err := t.first.ForwardOffline(stacked, rows)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	n := rows * projectedWidth
	a, err := ptxCleanupMalloc(n)
	if err != nil {
		return nil, err
	}
	owners := []*ptx.Buffer{a}
	// Keep original owners independent of the a/b ping-pong references.
	// CUDA work may still be queued if a later launch or context check fails.
	defer func() {
		err = errors.Join(err, t.releaseForwardOwners(owners))
		if err != nil {
			result = nil
		}
	}()
	b, err := ptxCleanupMalloc(n)
	if err != nil {
		return nil, err
	}
	owners = append(owners, b)
	if err := a.Upload(first); err != nil {
		return nil, err
	}
	for index, layer := range t.layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := layer.forwardWithScratch(b, a, rows, t.scratch); err != nil {
			return nil, fmt.Errorf("PTX tower layer %d: %w", index+1, err)
		}
		a, b = b, a
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := ptx.AffineLayerNormF32Buffer(b, a, t.weight, t.bias, rows, projectedWidth, 1e-5); err != nil {
		return nil, err
	}
	if err := ptxCleanupSync(); err != nil {
		t.terminal = fmt.Errorf("PTX tower forward sync: %w", err)
		return nil, t.terminal
	}
	out := make([]float32, n)
	if err := b.Download(out); err != nil {
		return nil, err
	}
	for _, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite PTX diarization tower output")
		}
	}
	return out, nil
}
