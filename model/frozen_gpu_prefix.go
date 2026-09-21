package model

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"time"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// FrozenPrefix is immutable per-layer, post-RoPE K/V for an exact token prefix.
// One live prefix per encoder; callers cannot supply or mutate its device state.
type FrozenPrefix struct {
	owner        *FrozenGPUEncoder
	ids          []int
	keys, values []*nvidia.DevBuf
	workK, workV *nvidia.DevBuf
	bytes        uint64
	closed       bool
}
type PrefixBranch struct {
	ID string
	// ValidSuffixTokens must equal len(Suffix): this API accepts ragged real
	// tokens, never padded matrices. A nonmatching declared length is rejected.
	ValidSuffixTokens int
	Suffix            []int
	CandidateIDs      []string
	CandidateTokens   []int
}
type PrefixResult struct {
	ID            string    `json:"id"`
	CandidateIDs  []string  `json:"candidate_ids"`
	Logits        []float32 `json:"logits"`
	LastRealToken int       `json:"last_real_token"`
}
type PrefixTiming struct {
	QueueSeconds      float64 `json:"queue_seconds"`
	UploadSeconds     float64 `json:"upload_seconds"`
	SuffixSeconds     float64 `json:"suffix_seconds"`
	DownloadSeconds   float64 `json:"download_seconds"`
	ProjectionSeconds float64 `json:"projection_seconds"`
	UploadBytes       int     `json:"upload_bytes"`
	DownloadBytes     int     `json:"download_bytes"`
	DeviceCopyBytes   uint64  `json:"device_copy_bytes"`
	PrefixBytes       uint64  `json:"prefix_bytes"`
}
type prefixWork struct{ branches, suffix, effective, candidates int }

func (e *FrozenGPUEncoder) reservePrefixWork(w prefixWork) (func(), error) {
	e.prefixAdmissionMu.Lock()
	defer e.prefixAdmissionMu.Unlock()
	q := e.prefixQueued
	if w.branches+q.branches > 32 || w.suffix+q.suffix > 2048 || w.effective+q.effective > 4096 || w.candidates+q.candidates > 512 {
		return nil, fmt.Errorf("prefix queue token/branch/candidate budget exceeded")
	}
	e.prefixQueued = prefixWork{q.branches + w.branches, q.suffix + w.suffix, q.effective + w.effective, q.candidates + w.candidates}
	var once sync.Once
	return func() {
		once.Do(func() {
			e.prefixAdmissionMu.Lock()
			defer e.prefixAdmissionMu.Unlock()
			e.prefixQueued.branches -= w.branches
			e.prefixQueued.suffix -= w.suffix
			e.prefixQueued.effective -= w.effective
			e.prefixQueued.candidates -= w.candidates
		})
	}, nil
}
func validatePrefixBranches(prefixLen, maxTokens, vocab int, branches []PrefixBranch) (prefixWork, error) {
	var w prefixWork
	if prefixLen < 1 || prefixLen >= maxTokens || len(branches) < 1 || len(branches) > 8 {
		return w, fmt.Errorf("prefix/branch count outside bounds")
	}
	seen := map[string]bool{}
	for _, b := range branches {
		if b.ID == "" || seen[b.ID] || len(b.Suffix) < 1 || b.ValidSuffixTokens != len(b.Suffix) || len(b.Suffix)+prefixLen > maxTokens || len(b.CandidateIDs) != len(b.CandidateTokens) || len(b.CandidateTokens) < 2 || len(b.CandidateTokens) > 32 {
			return w, fmt.Errorf("invalid branch shape/identity")
		}
		seen[b.ID] = true
		for _, id := range b.Suffix {
			if id < 0 || id >= vocab {
				return w, fmt.Errorf("invalid suffix token")
			}
		}
		ids := map[string]bool{}
		tokens := map[int]bool{}
		for i, id := range b.CandidateIDs {
			tok := b.CandidateTokens[i]
			if id == "" || ids[id] || tok < 0 || tok >= vocab || tokens[tok] {
				return w, fmt.Errorf("invalid candidate identity/token")
			}
			ids[id] = true
			tokens[tok] = true
		}
		w.branches++
		w.suffix += len(b.Suffix)
		w.effective += prefixLen + len(b.Suffix)
		w.candidates += len(b.CandidateTokens)
	}
	if w.suffix > maxTokens || w.effective > 2048 || w.candidates > 128 {
		return w, fmt.Errorf("aggregate suffix/effective/candidate budget exceeded")
	}
	return w, nil
}
func (e *FrozenGPUEncoder) NewFrozenPrefix(ctx context.Context, ids []int) (p *FrozenPrefix, err error) {
	if e == nil || ctx == nil {
		return nil, fmt.Errorf("encoder/context required")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return nil, fmt.Errorf("encoder closed")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if err = validateFrozenGPURequest(e.cfg, e.maxTokens, ids); err != nil {
		return nil, err
	}
	if len(ids) >= e.maxTokens {
		return nil, fmt.Errorf("prefix leaves no suffix space")
	}
	if e.prefix != nil {
		return nil, fmt.Errorf("close existing prefix before replacement")
	}
	k := e.cfg.NumKVHeads * e.cfg.HeadDim
	n := len(ids)
	bytes := uint64((len(e.layers)*n + e.maxTokens) * k * 2 * 4)
	free, _ := nvidia.MemInfo()
	if bytes > e.stats.BudgetBytes-e.stats.AllocatedBytes || free < bytes+e.stats.ReserveBytes {
		return nil, fmt.Errorf("prefix GPU budget/reserve exceeded")
	}
	p = &FrozenPrefix{owner: e, ids: append([]int(nil), ids...), bytes: bytes}
	owned := p
	defer func() {
		if err != nil {
			nvidia.SyncAll()
			owned.freeLocked()
			p = nil
		}
	}()
	for i := 0; i < len(e.layers); i++ {
		a, x := nvidia.NewDevBufGPU(n * k)
		if x != nil {
			return nil, x
		}
		p.keys = append(p.keys, a)
		b, x := nvidia.NewDevBufGPU(n * k)
		if x != nil {
			return nil, x
		}
		p.values = append(p.values, b)
	}
	p.workK, err = nvidia.NewDevBufGPU(e.maxTokens * k)
	if err != nil {
		return nil, err
	}
	p.workV, err = nvidia.NewDevBufGPU(e.maxTokens * k)
	if err != nil {
		return nil, err
	}
	for pos, id := range ids {
		if err = e.uploadEmbedding(id, pos); err != nil {
			return nil, err
		}
	}
	for layer := range e.layers {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		err = e.forwardLayerAttention(layer, n, func() error {
			if x := e.independentFrozenAttention(layer, n); x != nil {
				return x
			}
			if x := frozenGPUCopy(p.keys[layer], e.kNormed, n*k); x != nil {
				return x
			}
			return frozenGPUCopy(p.values[layer], e.v, n*k)
		})
		if err != nil {
			return nil, err
		}
	}
	if err = nvidia.SyncErr(); err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	e.prefix = p
	return p, nil
}
func (p *FrozenPrefix) freeLocked() {
	if p == nil || p.closed {
		return
	}
	for _, b := range p.keys {
		b.Free()
	}
	for _, b := range p.values {
		b.Free()
	}
	if p.workK != nil {
		p.workK.Free()
	}
	if p.workV != nil {
		p.workV.Free()
	}
	p.closed = true
	p.keys = nil
	p.values = nil
}
func (e *FrozenGPUEncoder) closePrefixLocked() {
	if e.prefix != nil {
		nvidia.SyncAll()
		e.prefix.freeLocked()
		e.prefix = nil
	}
}
func (p *FrozenPrefix) Close() {
	if p == nil || p.owner == nil {
		return
	}
	e := p.owner
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.prefix == p {
		e.closePrefixLocked()
	} else {
		p.freeLocked()
	}
}
func (p *FrozenPrefix) Bytes() uint64 {
	if p == nil {
		return 0
	}
	return p.bytes
}

// Fingerprint is a diagnostic device-to-host checksum, not part of scoring.
func (p *FrozenPrefix) Fingerprint() (string, error) {
	if p == nil || p.owner == nil {
		return "", fmt.Errorf("nil prefix")
	}
	e := p.owner
	e.mu.Lock()
	defer e.mu.Unlock()
	if p.closed || e.closed {
		return "", fmt.Errorf("prefix closed")
	}
	h := sha256.New()
	for _, list := range [][]*nvidia.DevBuf{p.keys, p.values} {
		for _, b := range list {
			v := make([]float32, len(p.ids)*e.cfg.NumKVHeads*e.cfg.HeadDim)
			if err := b.GPUBuffer().Download(v); err != nil {
				return "", err
			}
			data := make([]byte, len(v)*4)
			for i, x := range v {
				binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(x))
			}
			h.Write(data)
		}
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// ScoreSuffixes uses ragged packed projections and isolated causal attention.
// packed=false executes each suffix serially. It never pads or appends siblings
// to prefix state. Cancellation drains already submitted kernels before return.
func (p *FrozenPrefix) ScoreSuffixes(ctx context.Context, branches []PrefixBranch, packed bool) ([]PrefixResult, PrefixTiming, error) {
	var timing PrefixTiming
	if p == nil || p.owner == nil || ctx == nil {
		return nil, timing, fmt.Errorf("prefix/context required")
	}
	e := p.owner
	w, err := validatePrefixBranches(len(p.ids), e.maxTokens, e.cfg.VocabSize, branches)
	if err != nil {
		return nil, timing, err
	}
	if err = ctx.Err(); err != nil {
		return nil, timing, err
	}
	release, err := e.reservePrefixWork(w)
	if err != nil {
		return nil, timing, err
	}
	defer release()
	e.prefixGateOnce.Do(func() { e.prefixGate = make(chan struct{}, 1) })
	queued := time.Now()
	select {
	case e.prefixGate <- struct{}{}:
	case <-ctx.Done():
		return nil, timing, ctx.Err()
	}
	defer func() { <-e.prefixGate }()
	e.mu.Lock()
	defer e.mu.Unlock()
	defer nvidia.SyncAll()
	timing.QueueSeconds = time.Since(queued).Seconds()
	timing.PrefixBytes = p.bytes
	if p.closed || e.closed || e.prefix != p {
		return nil, timing, fmt.Errorf("stale prefix")
	}
	if err = ctx.Err(); err != nil {
		return nil, timing, err
	}
	// Copy caller inputs before use; caller must not mutate them during the call.
	owned := make([]PrefixBranch, len(branches))
	for i, b := range branches {
		owned[i] = PrefixBranch{ID: b.ID, ValidSuffixTokens: b.ValidSuffixTokens, Suffix: append([]int(nil), b.Suffix...), CandidateIDs: append([]string(nil), b.CandidateIDs...), CandidateTokens: append([]int(nil), b.CandidateTokens...)}
	}
	var results []PrefixResult
	if packed {
		results, err = p.scorePackedLocked(ctx, owned, &timing)
	} else {
		for _, b := range owned {
			r, x := p.scorePackedLocked(ctx, []PrefixBranch{b}, &timing)
			if x != nil {
				err = x
				break
			}
			results = append(results, r...)
		}
	}
	if err == nil {
		err = ctx.Err()
	}
	if err != nil {
		return nil, timing, err
	}
	return results, timing, nil
}
func (p *FrozenPrefix) scorePackedLocked(ctx context.Context, branches []PrefixBranch, t *PrefixTiming) ([]PrefixResult, error) {
	e := p.owner
	h := e.cfg.HiddenSize
	k := e.cfg.NumKVHeads * e.cfg.HeadDim
	prefix := len(p.ids)
	offset := 0
	start := time.Now()
	for _, b := range branches {
		for _, id := range b.Suffix {
			if err := e.uploadEmbedding(id, offset); err != nil {
				return nil, err
			}
			offset++
		}
	}
	if err := nvidia.SyncErr(); err != nil {
		return nil, err
	}
	t.UploadSeconds += time.Since(start).Seconds()
	t.UploadBytes += offset * h * 4
	start = time.Now()
	for layer := range e.layers {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		l := &e.layers[layer]
		q := l.qDim
		err := e.forwardLayerAttention(layer, offset, func() error {
			off := 0
			for _, b := range branches {
				n := len(b.Suffix)
				if err := frozenGPUCopy(p.workK, p.keys[layer], prefix*k); err != nil {
					return err
				}
				if err := frozenGPUCopy(p.workV, p.values[layer], prefix*k); err != nil {
					return err
				}
				for j := 0; j < n; j++ {
					if !nvidia.DevRoPE(e.qNormed.Slice((off+j)*q, q), e.ropeTable, prefix+j, e.cfg.NumHeads, l.headDim) || !nvidia.DevRoPE(e.kNormed.Slice((off+j)*k, k), e.ropeTable, prefix+j, e.cfg.NumKVHeads, l.headDim) {
						return fmt.Errorf("prefix RoPE rejected")
					}
				}
				if err := frozenGPUCopy(p.workK.Slice(prefix*k, n*k), e.kNormed.Slice(off*k, n*k), n*k); err != nil {
					return err
				}
				if err := frozenGPUCopy(p.workV.Slice(prefix*k, n*k), e.v.Slice(off*k, n*k), n*k); err != nil {
					return err
				}
				t.DeviceCopyBytes += uint64((prefix + n) * k * 2 * 4)
				for j := 0; j < n; j++ {
					keys := prefix + j + 1
					if !nvidia.DevAttentionOK(e.attnOut.Slice((off+j)*q, q), e.qNormed.Slice((off+j)*q, q), p.workK.Slice(0, keys*k), p.workV.Slice(0, keys*k), keys, e.cfg.NumHeads, e.cfg.NumKVHeads, l.headDim, attentionScale(e.cfg, l.headDim)) {
						return fmt.Errorf("prefix attention rejected")
					}
				}
				off += n
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	// Terminal scoring normalises only each last real token, never vocabulary rows.
	off := 0
	for _, b := range branches {
		pos := off + len(b.Suffix) - 1
		if err := e.normRows(e.normed.Slice(pos*h, h), e.hidden.Slice(pos*h, h), e.finalNorm, 1, h); err != nil {
			return nil, err
		}
		off += len(b.Suffix)
	}
	if err := nvidia.SyncErr(); err != nil {
		return nil, err
	}
	t.SuffixSeconds += time.Since(start).Seconds()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	var results []PrefixResult
	off = 0
	for _, b := range branches {
		pos := off + len(b.Suffix) - 1
		hidden := make([]float32, h)
		start = time.Now()
		if err := e.normed.Slice(pos*h, h).GPUBuffer().Download(hidden); err != nil {
			return nil, err
		}
		t.DownloadSeconds += time.Since(start).Seconds()
		t.DownloadBytes += h * 4
		start = time.Now()
		raw, bias, err := e.selectedHeadLocked(b.CandidateTokens)
		if err != nil {
			return nil, err
		}
		logits, err := selectedBF16Logits(hidden, raw, e.cfg.VocabSize, b.CandidateTokens, bias)
		if err != nil {
			return nil, err
		}
		t.ProjectionSeconds += time.Since(start).Seconds()
		results = append(results, PrefixResult{b.ID, b.CandidateIDs, logits, prefix + len(b.Suffix) - 1})
		off += len(b.Suffix)
	}
	return results, nil
}
