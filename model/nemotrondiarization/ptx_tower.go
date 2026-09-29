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
	first        *Layer0Complete
	layers       []*PTXAudioLayer
	weight, bias *ptx.Buffer
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
			t.Close()
		}
	}()
	for index, model := range source.remaining {
		var layer *PTXAudioLayer
		layer, err = NewPTXAudioLayer(model, maxRows)
		if err != nil {
			return nil, fmt.Errorf("PTX tower layer %d: %w", index+1, err)
		}
		t.layers = append(t.layers, layer)
	}
	upload := func(data []float32) (*ptx.Buffer, error) {
		b, e := ptx.Malloc(len(data))
		if e != nil {
			return nil, e
		}
		if e = b.Upload(data); e != nil {
			b.Free()
			return nil, e
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
	return t, nil
}

// Close is idempotent and waits for an active ForwardRows call.
func (t *PTXAudioTower) Close() {
	if t == nil {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.closed {
		return
	}
	t.closed = true
	if t.bias != nil {
		t.bias.Free()
	}
	if t.weight != nil {
		t.weight.Free()
	}
	for i := len(t.layers) - 1; i >= 0; i-- {
		t.layers[i].Close()
	}
	t.layers, t.first, t.weight, t.bias = nil, nil, nil, nil
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
	if t.closed || t.first == nil || len(t.layers) != 30 || rows < 1 || rows > t.maxRows || len(stacked) != rows*projectedWidth {
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
	a, err := ptx.Malloc(n)
	if err != nil {
		return nil, err
	}
	owners := []*ptx.Buffer{a}
	// Keep original owners independent of the a/b ping-pong references.
	// CUDA work may still be queued if a later launch or context check fails.
	defer func() {
		err = errors.Join(err, ptx.SyncErr())
		for i := len(owners) - 1; i >= 0; i-- {
			owners[i].Free()
		}
		if err != nil {
			result = nil
		}
	}()
	b, err := ptx.Malloc(n)
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
		if err := layer.ForwardBuffer(b, a, rows); err != nil {
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
	if err := ptx.SyncErr(); err != nil {
		return nil, err
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
