package model

import (
	"context"
	"fmt"
	"math"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
)

// Gemma4PackedRows bounds total real token rows in a packed transformer pass.
const Gemma4PackedRows = 512

type gemma4PrefillSegment struct {
	offset, length, parentStart, parentLength int
	terminal                                  bool
}

// Reused only while the owning Gemma4NVIDIA mutex is held. Each slot grows to
// at most the 512-row workspace; Close releases every retained allocation.
type gemma4DeviceWork struct {
	buffers []*nvidia.Buffer
	next    int
	err     error
}

func (w *gemma4DeviceWork) begin() { w.next, w.err = 0, nil }

func (w *gemma4DeviceWork) alloc(elements int) *nvidia.Buffer {
	if w.err != nil {
		return nil
	}
	index := w.next
	w.next++
	if index == len(w.buffers) {
		w.buffers = append(w.buffers, nil)
	}
	b := w.buffers[index]
	if b != nil && elements <= b.Size/4 {
		return b
	}
	if b != nil {
		b.Free()
		w.buffers[index] = nil
	}
	b, w.err = nvidia.Malloc(elements)
	if w.err == nil {
		w.buffers[index] = b
	}
	return b
}

func (w *gemma4DeviceWork) free() {
	for _, b := range w.buffers {
		if b != nil {
			b.Free()
		}
	}
	w.buffers = nil
	w.begin()
}

// deviceView borrows storage; the returned view must never be freed.
func deviceView(b *nvidia.Buffer, offset, count int) *nvidia.Buffer {
	return &nvidia.Buffer{Ptr: b.Ptr + nvidia.CUdeviceptr(offset*4), Size: count * 4}
}

func (g *Gemma4NVIDIA) prefillAttentionSegments(out, q, k, v, rope *nvidia.Buffer, arena *gemma4NVIDIAKVArena, layer, rows, pos0, window, hd, rot int, packed *nvidia.SegmentedRows) error {
	m := g.model
	kvHeads := gemmacfg.LayerKVHeads(m.Config, layer)
	if packed != nil {
		if err := packed.RoPE(q, rope, m.Config.NumHeads, hd, rot); err != nil {
			return err
		}
		if err := packed.RoPE(k, rope, kvHeads, hd, rot); err != nil {
			return err
		}
		return packed.AttentionFromPrefix(out, q, k, v, arena.trunkK[layer], arena.trunkV[layer], arena.trunkBase[layer], window, m.Config.NumHeads, kvHeads, hd, attentionScale(m.Config, hd))
	}
	if err := nvidia.RoPEPartialSequenceBuffer(q, rope, rows, pos0, m.Config.NumHeads, hd, rot); err != nil {
		return err
	}
	if err := nvidia.RoPEPartialSequenceBuffer(k, rope, rows, pos0, kvHeads, hd, rot); err != nil {
		return err
	}
	if err := arena.appendTrunkRows(layer, pos0, rows, k, v); err != nil {
		return err
	}
	base := arena.trunkBase[layer]
	return nvidia.CausalBatchAttentionBuffer(out, q, arena.trunkK[layer], arena.trunkV[layer], rows, pos0-base, pos0+rows-base, window, m.Config.NumHeads, kvHeads, hd, attentionScale(m.Config, hd))
}

// ScorePrefixedPacked scores independent token sequences after one immutable
// prefix. Projections run across all real token rows; attention stays isolated.
// The caller chunks to Gemma4PackedRows and owns the prefix lifetime.
func (g *Gemma4NVIDIA) ScorePrefixedPacked(ctx context.Context, prefix *Gemma4NVIDIAContext, paths, candidates [][]int) ([][]float32, error) {
	if g == nil || ctx == nil {
		return nil, fmt.Errorf("nil NVIDIA packed scorer/context")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed || prefix == nil || prefix.closed || prefix.owner != g || prefix.arena == nil {
		return nil, fmt.Errorf("invalid NVIDIA packed prefix")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(paths) == 0 || len(paths) > 256 || len(paths) != len(candidates) {
		return nil, fmt.Errorf("invalid NVIDIA packed paths/candidates")
	}
	segments := make([]gemma4PrefillSegment, len(paths))
	rows := 0
	for i, path := range paths {
		if len(path) == 0 || len(path) > Gemma4PackedRows-rows || len(candidates[i]) == 0 || len(candidates[i]) > 255 || len(prefix.tokens)+len(path) > g.contextLimit() {
			return nil, fmt.Errorf("invalid NVIDIA packed path %d", i)
		}
		for _, ids := range [][]int{path, candidates[i]} {
			for _, id := range ids {
				if id < 0 || id >= g.model.Config.VocabSize {
					return nil, fmt.Errorf("packed path %d token outside vocabulary", i)
				}
			}
		}
		segments[i] = gemma4PrefillSegment{offset: rows, length: len(path), terminal: true}
		rows += len(path)
	}
	flat := make([]int, 0, rows)
	for _, path := range paths {
		flat = append(flat, path...)
	}
	return g.scorePackedPlan(ctx, prefix, flat, segments, candidates)
}

// Caller holds g.mu and validates the plan before entering.
func (g *Gemma4NVIDIA) scorePackedPlan(ctx context.Context, prefix *Gemma4NVIDIAContext, flat []int, segments []gemma4PrefillSegment, candidates [][]int) ([][]float32, error) {
	h := g.model.Config.HiddenSize
	terminal, err := nvidia.Malloc(len(candidates) * h)
	if err != nil {
		return nil, err
	}
	defer terminal.Free()
	// Drain submitted work on cancellation/error before releasing owned buffers.
	defer nvidia.SyncAll()
	if err := g.runPrefillRows(ctx, flat, len(prefix.tokens), prefix.arena, nil, terminal, segments); err != nil {
		return nil, err
	}
	if err := nvidia.IdeogramRMSNormRowsBuffer(terminal, terminal, g.norm, nil, len(candidates), h, float32(g.model.Config.RMSNormEps), false); err != nil {
		return nil, err
	}
	out, err := g.lmHead.ProjectSelectedBatch(terminal, candidates)
	if err != nil {
		return nil, err
	}
	for i := range out {
		g.transformSelectedLogits(out[i], candidates[i])
	}
	return out, ctx.Err()
}

func (g *Gemma4NVIDIA) transformSelectedLogits(logits []float32, tokens []int) {
	applyLlamaFinalLogitSoftcap(logits, g.model.Config.FinalLogitSoftcapping)
	for i, token := range tokens {
		for _, suppressed := range g.model.SuppressTokens {
			if token == suppressed {
				logits[i] = float32(math.Inf(-1))
				break
			}
		}
	}
}

// ScorePrefixedTreeGroups shares each context once among its independent field
// branches, and packs multiple contexts into the same transformer projections.
func (g *Gemma4NVIDIA) ScorePrefixedTreeGroups(ctx context.Context, prefix *Gemma4NVIDIAContext, contexts, branches, candidates [][]int) ([][][]float32, error) {
	if g == nil || ctx == nil {
		return nil, fmt.Errorf("nil tree scorer/context")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed || prefix == nil || prefix.closed || prefix.owner != g || prefix.arena == nil {
		return nil, fmt.Errorf("invalid tree prefix")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(contexts) < 1 || len(branches) < 1 || len(branches) != len(candidates) || len(contexts) > 256/len(branches) {
		return nil, fmt.Errorf("invalid tree group shape")
	}
	var flat []int
	var segments []gemma4PrefillSegment
	var selected [][]int
	appendTokens := func(tokens []int) error {
		if len(tokens) < 1 || len(tokens) > Gemma4PackedRows-len(flat) {
			return fmt.Errorf("tree group token budget exceeded")
		}
		for _, id := range tokens {
			if id < 0 || id >= g.model.Config.VocabSize {
				return fmt.Errorf("tree group token outside vocabulary")
			}
		}
		flat = append(flat, tokens...)
		return nil
	}
	for _, c := range contexts {
		start := len(flat)
		if err := appendTokens(c); err != nil {
			return nil, err
		}
		segments = append(segments, gemma4PrefillSegment{offset: start, length: len(c)})
		for j, b := range branches {
			if len(prefix.tokens)+len(c)+len(b) > g.contextLimit() || len(candidates[j]) < 1 || len(candidates[j]) > 255 {
				return nil, fmt.Errorf("invalid tree group branch")
			}
			for _, id := range candidates[j] {
				if id < 0 || id >= g.model.Config.VocabSize {
					return nil, fmt.Errorf("candidate outside vocabulary")
				}
			}
			at := len(flat)
			if err := appendTokens(b); err != nil {
				return nil, err
			}
			segments = append(segments, gemma4PrefillSegment{offset: at, length: len(b), parentStart: start, parentLength: len(c), terminal: true})
			selected = append(selected, candidates[j])
		}
	}
	logits, err := g.scorePackedPlan(ctx, prefix, flat, segments, selected)
	if err != nil {
		return nil, err
	}
	out := make([][][]float32, len(contexts))
	for i := range out {
		out[i] = logits[i*len(branches) : (i+1)*len(branches)]
	}
	return out, nil
}
