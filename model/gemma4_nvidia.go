package model

import (
	"context"
	"fmt"
	"sync"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/loader/gguf"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
)

// Gemma4NVIDIA owns resident PTX projections for one model. Requests are
// serialized because the scratch/KV arena is reused; model weights remain
// resident across calls. It never substitutes CPU transformer execution.
type Gemma4NVIDIA struct {
	model       *LlamaModel
	mu          sync.Mutex
	layers      []gemma4NVIDIALayer
	norm        *nvidia.Buffer
	lmHead      *nvidia.GPUGGUFMatrix
	ropeSWA     *nvidia.Buffer
	ropeFull    *nvidia.Buffer
	closed      bool
	prefillWork gemma4DeviceWork
}

type gemma4NVIDIALayer struct {
	q, k, v, o                                   *nvidia.GPUGGUFMatrix
	gate, up, gateUp, down                       *nvidia.GPUGGUFMatrix
	inputNorm, postNorm, preFFNNorm, postFFNNorm *nvidia.Buffer
	qNorm, kNorm                                 *nvidia.Buffer
}

func q4GateUpCompatible(gate, up *gguf.QuantMatrix) bool {
	return gate != nil && up != nil && gate.QType == gguf.QuantQ4_K && up.QType == gguf.QuantQ4_K && gate.InDim == up.InDim && gate.OutDim == up.OutDim
}

func NewGemma4NVIDIA(m *LlamaModel) (*Gemma4NVIDIA, error) {
	if m == nil || m.Config.ModelType != "gemma4_text" {
		return nil, fmt.Errorf("NVIDIA Gemma4 requires gemma4_text model")
	}
	if !nvidia.SgemmReady() {
		if nvidia.Available() {
			return nil, fmt.Errorf("CUDA device available but PTX runtime not ready")
		}
		return nil, fmt.Errorf("CUDA device unavailable")
	}
	if err := m.validateGemma4IndependentBranchLayers(); err != nil {
		return nil, err
	}
	if m.Config.HiddenPerLayer != 0 || m.PerLayerModelProj != nil {
		return nil, fmt.Errorf("NVIDIA Gemma4 PLI graph is not implemented")
	}
	g := &Gemma4NVIDIA{model: m, layers: make([]gemma4NVIDIALayer, m.Config.NumLayers)}
	ok := false
	defer func() {
		if !ok {
			g.Close()
		}
	}()
	uploadF32 := func(data []float32) (*nvidia.Buffer, error) {
		b, err := nvidia.Malloc(len(data))
		if err != nil {
			return nil, err
		}
		if err = b.Upload(data); err != nil {
			b.Free()
			return nil, err
		}
		return b, nil
	}
	var err error
	for i := range g.layers {
		src := &m.Layers[i]
		dst := &g.layers[i]
		if dst.q, err = nvidia.UploadGGUFMatrix(src.QWGGUF); err != nil {
			return nil, fmt.Errorf("layer %d Q: %w", i, err)
		}
		if src.HasKV {
			if dst.k, err = nvidia.UploadGGUFMatrix(src.KWGGUF); err != nil {
				return nil, fmt.Errorf("layer %d K: %w", i, err)
			}
			if src.VWGGUF == src.KWGGUF {
				dst.v = dst.k
			} else if dst.v, err = nvidia.UploadGGUFMatrix(src.VWGGUF); err != nil {
				return nil, fmt.Errorf("layer %d V: %w", i, err)
			}
		}
		for name, pair := range map[string]struct {
			src *gguf.QuantMatrix
			dst **nvidia.GPUGGUFMatrix
		}{"O": {src.OWGGUF, &dst.o}, "down": {src.DownWGGUF, &dst.down}} {
			if *pair.dst, err = nvidia.UploadGGUFMatrix(pair.src); err != nil {
				return nil, fmt.Errorf("layer %d %s: %w", i, name, err)
			}
		}
		if q4GateUpCompatible(src.GateWGGUF, src.UpWGGUF) {
			raw := make([]byte, 0, len(src.GateWGGUF.Raw)+len(src.UpWGGUF.Raw))
			raw = append(raw, src.GateWGGUF.Raw...)
			raw = append(raw, src.UpWGGUF.Raw...)
			combined := &gguf.QuantMatrix{Name: src.GateWGGUF.Name + "+" + src.UpWGGUF.Name, QType: gguf.QuantQ4_K, Raw: raw, InDim: src.GateWGGUF.InDim, OutDim: src.GateWGGUF.OutDim + src.UpWGGUF.OutDim}
			if dst.gateUp, err = nvidia.UploadGGUFMatrix(combined); err != nil {
				return nil, fmt.Errorf("layer %d gate/up: %w", i, err)
			}
		} else {
			if dst.gate, err = nvidia.UploadGGUFMatrix(src.GateWGGUF); err != nil {
				return nil, fmt.Errorf("layer %d gate: %w", i, err)
			}
			if dst.up, err = nvidia.UploadGGUFMatrix(src.UpWGGUF); err != nil {
				return nil, fmt.Errorf("layer %d up: %w", i, err)
			}
		}
		if dst.inputNorm, err = uploadF32(src.InputNorm.Data()); err != nil {
			return nil, err
		}
		if dst.postNorm, err = uploadF32(src.PostNorm.Data()); err != nil {
			return nil, err
		}
		if src.PreFFNNorm != nil {
			if dst.preFFNNorm, err = uploadF32(src.PreFFNNorm.Data()); err != nil {
				return nil, err
			}
		}
		if src.PostFFNNorm != nil {
			if dst.postFFNNorm, err = uploadF32(src.PostFFNNorm.Data()); err != nil {
				return nil, err
			}
		}
		if src.QNorm != nil {
			if dst.qNorm, err = uploadF32(src.QNorm.Data()); err != nil {
				return nil, err
			}
		}
		if src.KNorm != nil {
			if dst.kNorm, err = uploadF32(src.KNorm.Data()); err != nil {
				return nil, err
			}
		}
	}
	if g.norm, err = uploadF32(m.Norm.Data()); err != nil {
		return nil, err
	}
	if g.lmHead, err = nvidia.UploadGGUFMatrix(m.LMHeadGGUF); err != nil {
		return nil, fmt.Errorf("LM head: %w", err)
	}
	maxSeq := m.Config.MaxSeqLen
	if maxSeq <= 0 || maxSeq > 2048 {
		maxSeq = 2048
	}
	ropeCfg := m.Config
	if ropeCfg.GlobalHeadDim <= 0 {
		ropeCfg.GlobalHeadDim = ropeCfg.HeadDim
	}
	swa, _, full, _ := gemma4RoPETables(ropeCfg, maxSeq, nil)
	if len(swa) > 0 {
		if g.ropeSWA, err = uploadF32(swa); err != nil {
			return nil, err
		}
	}
	if len(full) > 0 {
		if g.ropeFull, err = uploadF32(full); err != nil {
			return nil, err
		}
	}
	ok = true
	return g, nil
}

func (g *Gemma4NVIDIA) Close() {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return
	}
	g.closed = true
	nvidia.SyncAll()
	g.prefillWork.free()
	for i := range g.layers {
		g.layers[i].free()
	}
	if g.norm != nil {
		g.norm.Free()
		g.norm = nil
	}
	if g.lmHead != nil {
		g.lmHead.Free()
		g.lmHead = nil
	}
	if g.ropeSWA != nil {
		g.ropeSWA.Free()
		g.ropeSWA = nil
	}
	if g.ropeFull != nil {
		g.ropeFull.Free()
		g.ropeFull = nil
	}
}
func (l *gemma4NVIDIALayer) free() {
	if l == nil {
		return
	}
	seen := map[*nvidia.GPUGGUFMatrix]bool{}
	for _, m := range []*nvidia.GPUGGUFMatrix{l.q, l.k, l.v, l.o, l.gate, l.up, l.gateUp, l.down} {
		if m != nil && !seen[m] {
			m.Free()
			seen[m] = true
		}
	}
	for _, b := range []*nvidia.Buffer{l.inputNorm, l.postNorm, l.preFFNNorm, l.postFFNNorm, l.qNorm, l.kNorm} {
		if b != nil {
			b.Free()
		}
	}
}

// ScoreIndependentBranches runs the full transformer and LM head on PTX. The
// CPU supplies token embeddings and immutable prefilled trunk KV only.
// PrefillPrepared executes an already templated prompt on the resident PTX
// graph and returns request-owned device KV. The caller must close the context.
func (g *Gemma4NVIDIA) PrefillPrepared(ctx context.Context, tokens []int, branchCap, suffixCap int) (*Gemma4NVIDIAContext, error) {
	return g.PrefillPreparedCapacity(ctx, tokens, len(tokens), branchCap, suffixCap)
}

func (g *Gemma4NVIDIA) PrefillPreparedCapacity(ctx context.Context, tokens []int, trunkCap, branchCap, suffixCap int) (*Gemma4NVIDIAContext, error) {
	if g == nil || ctx == nil || len(tokens) == 0 || trunkCap < len(tokens) || trunkCap > 2048 || branchCap <= 0 || branchCap > 256 || suffixCap <= 0 {
		return nil, fmt.Errorf("invalid NVIDIA Gemma4 prefill")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return nil, fmt.Errorf("NVIDIA Gemma4 closed")
	}
	a, err := allocGemma4NVIDIAKVArena(g.model, trunkCap, branchCap, suffixCap)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			a.free()
		}
	}()
	for start := 0; start < len(tokens); start += 512 {
		end := min(start+512, len(tokens))
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := g.runPrefillBatch(ctx, tokens[start:end], start, a, nil); err != nil {
			return nil, fmt.Errorf("prompt ubatch [%d,%d): %w", start, end, err)
		}
	}
	a.trunkUsed = len(tokens)
	ok = true
	return &Gemma4NVIDIAContext{owner: g, tokens: append([]int(nil), tokens...), arena: a}, nil
}

func (g *Gemma4NVIDIA) runPrefillBatch(ctx context.Context, tokens []int, pos0 int, arena *gemma4NVIDIAKVArena, final []float32) error {
	return g.runPrefillBatchOutput(ctx, tokens, pos0, arena, final, nil)
}
func (g *Gemma4NVIDIA) runPrefillBatchOutput(ctx context.Context, tokens []int, pos0 int, arena *gemma4NVIDIAKVArena, final []float32, finalDevice *nvidia.Buffer) error {
	return g.runPrefillRows(ctx, tokens, pos0, arena, final, finalDevice, nil)
}

// segments partitions packed projection rows into independent causal sequences.
// A nil partition retains the single-sequence prefill path.
func (g *Gemma4NVIDIA) runPrefillRows(ctx context.Context, tokens []int, pos0 int, arena *gemma4NVIDIAKVArena, final []float32, finalDevice *nvidia.Buffer, segments []gemma4PrefillSegment) error {
	m := g.model
	B, h := len(tokens), m.Config.HiddenSize
	if B < 1 || B > Gemma4PackedRows {
		return fmt.Errorf("prefill rows must be 1..%d", Gemma4PackedRows)
	}
	host := make([]float32, B*h)
	for i, tok := range tokens {
		if tok < 0 || tok >= m.Config.VocabSize {
			return fmt.Errorf("token %d outside vocab", tok)
		}
		if err := m.ScaledTokenEmbeddingInto(host[i*h:(i+1)*h], tok); err != nil {
			return err
		}
	}
	var packed *nvidia.SegmentedRows
	if len(segments) > 0 {
		layout := make([]nvidia.AttentionSegment, len(segments))
		for i, segment := range segments {
			layout[i] = nvidia.AttentionSegment{Length: segment.length, ParentStart: segment.parentStart, ParentLength: segment.parentLength}
		}
		var err error
		packed, err = nvidia.NewBranchedRows(layout, pos0)
		if err != nil {
			return err
		}
		defer packed.Close()
	}
	work := &g.prefillWork
	work.begin()
	defer nvidia.SyncAll()
	hidden := work.alloc(B * h)
	if work.err != nil {
		return work.err
	}
	var err error
	if err = hidden.Upload(host); err != nil {
		return err
	}
	residual := work.alloc(B * h)
	normed := work.alloc(B * h)
	maxQ, maxKV, maxInter := 1, 1, 1
	for l := range m.Layers {
		hd, _ := m.LayerHeadDim(l)
		maxQ = max(maxQ, m.Config.NumHeads*hd)
		maxKV = max(maxKV, gemmacfg.LayerKVHeads(m.Config, l)*hd)
		maxInter = max(maxInter, m.layerInterFor(&m.Layers[l]))
	}
	q := work.alloc(B * maxQ)
	k := work.alloc(B * maxKV)
	v := work.alloc(B * maxKV)
	attn := work.alloc(B * maxQ)
	o := work.alloc(B * h)
	gate := work.alloc(B * maxInter)
	up := work.alloc(B * maxInter)
	gateUp := work.alloc(B * maxInter * 2)
	down := work.alloc(B * h)
	if work.err != nil {
		return work.err
	}
	for l := 0; l < m.Config.NumLayers; l++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		src := &m.Layers[l]
		gl := &g.layers[l]
		hd, _ := m.LayerHeadDim(l)
		kvHeads := gemmacfg.LayerKVHeads(m.Config, l)
		kvDim := kvHeads * hd
		inter := m.layerInterFor(src)
		if err := nvidia.CopyDtoD(residual.Ptr, hidden.Ptr, uint64(B*h*4)); err != nil {
			return err
		}
		if err := nvidia.IdeogramRMSNormRowsBuffer(normed, hidden, gl.inputNorm, nil, B, h, float32(m.Config.RMSNormEps), false); err != nil {
			return err
		}
		if err := gl.q.ProjectBatchToBuffer(q, normed, B); err != nil {
			return err
		}
		if !src.HasKV {
			return fmt.Errorf("batched prefill requires materialized KV at layer %d", l)
		}
		if err := gl.k.ProjectBatchToBuffer(k, normed, B); err != nil {
			return err
		}
		if gl.v == gl.k {
			if err := nvidia.CopyDtoD(v.Ptr, k.Ptr, uint64(B*kvDim*4)); err != nil {
				return err
			}
		} else if err := gl.v.ProjectBatchToBuffer(v, normed, B); err != nil {
			return err
		}
		if gl.qNorm != nil {
			if err := nvidia.IdeogramRMSNormRowsBuffer(q, q, gl.qNorm, nil, B*m.Config.NumHeads, hd, float32(m.Config.RMSNormEps), false); err != nil {
				return err
			}
			if err := nvidia.IdeogramRMSNormRowsBuffer(k, k, gl.kNorm, nil, B*kvHeads, hd, float32(m.Config.RMSNormEps), false); err != nil {
				return err
			}
		}
		if err := nvidia.IdeogramRMSNormRowsNoWeightBuffer(v, v, B*kvHeads, hd, float32(m.Config.RMSNormEps)); err != nil {
			return err
		}
		lastPos := pos0 + B - 1
		if len(segments) > 0 {
			lastPos = pos0
			for _, segment := range segments {
				lastPos = max(lastPos, pos0+segment.parentLength+segment.length-1)
			}
		}
		_, rot := m.ensureGemma4RoPE(l, lastPos)
		rope := g.ropeSWA
		window := m.Config.SlidingWindow
		if len(m.Config.LayerTypes) <= l || m.Config.LayerTypes[l] != "sliding_attention" {
			rope = g.ropeFull
			window = 0
		}
		if err := g.prefillAttentionSegments(attn, q, k, v, rope, arena, l, B, pos0, window, hd, rot, packed); err != nil {
			return err
		}
		if err := gl.o.ProjectBatchToBuffer(o, attn, B); err != nil {
			return err
		}
		if gl.preFFNNorm == nil {
			return fmt.Errorf("layer %d missing pre-FFN norm", l)
		}
		if err := nvidia.IdeogramRMSNormRowsBuffer(o, o, gl.postNorm, nil, B, h, float32(m.Config.RMSNormEps), false); err != nil {
			return err
		}
		if err := nvidia.VecAddF32Buffer(residual, o, hidden, B*h); err != nil {
			return err
		}
		if err := nvidia.CopyDtoD(residual.Ptr, hidden.Ptr, uint64(B*h*4)); err != nil {
			return err
		}
		if err := nvidia.IdeogramRMSNormRowsBuffer(normed, hidden, gl.preFFNNorm, nil, B, h, float32(m.Config.RMSNormEps), false); err != nil {
			return err
		}
		if gl.gateUp != nil {
			if err := gl.gateUp.ProjectBatchToBuffer(gateUp, normed, B); err != nil {
				return err
			}
			if err := nvidia.GateUpGELUBuffer(gateUp, gate, B, inter); err != nil {
				return err
			}
		} else {
			if err := nvidia.ProjectQ4PairToBuffers(gate, up, normed, B, gl.gate, gl.up); err != nil {
				return err
			}
			if err := nvidia.GELUTanhMulBuffer(gate, up, B*inter); err != nil {
				return err
			}
		}
		if err := gl.down.ProjectBatchToBuffer(down, gate, B); err != nil {
			return err
		}
		if gl.postFFNNorm != nil {
			if err := nvidia.IdeogramRMSNormRowsBuffer(down, down, gl.postFFNNorm, nil, B, h, float32(m.Config.RMSNormEps), false); err != nil {
				return err
			}
		}
		if err := nvidia.VecAddF32Buffer(residual, down, hidden, B*h); err != nil {
			return err
		}
		if src.LayerScalar != 1 {
			if err := nvidia.VecScaleF32Buffer(hidden, hidden, B*h, src.LayerScalar); err != nil {
				return err
			}
		}
	}
	if finalDevice != nil {
		lastRows := []int{B - 1}
		if len(segments) > 0 {
			lastRows = nil
			for _, segment := range segments {
				if segment.terminal {
					lastRows = append(lastRows, segment.offset+segment.length-1)
				}
			}
		}
		if finalDevice.Size < len(lastRows)*h*4 {
			return fmt.Errorf("NVIDIA prefill final device buffer too small")
		}
		for i, lastRow := range lastRows {
			if err := nvidia.CopyDtoD(finalDevice.Ptr+nvidia.CUdeviceptr(i*h*4), hidden.Ptr+nvidia.CUdeviceptr(lastRow*h*4), uint64(h*4)); err != nil {
				return err
			}
		}
	}
	if final != nil {
		if len(final) < h {
			return fmt.Errorf("NVIDIA prefill final buffer=%d, want %d", len(final), h)
		}
		all := make([]float32, B*h)
		if err := hidden.Download(all); err != nil {
			return err
		}
		copy(final, all[(B-1)*h:B*h])
	}
	return nil
}

// ScorePrefixedSingleBranch packs a request context and one forced branch into
// a single causal prefill against an immutable cached prefix.
func (g *Gemma4NVIDIA) ScorePrefixedSingleBranch(ctx context.Context, prefix *Gemma4NVIDIAContext, tokens, candidates []int) ([]float32, error) {
	if g == nil || ctx == nil || prefix == nil || prefix.closed || prefix.owner != g || prefix.arena == nil || len(tokens) == 0 || len(prefix.tokens)+len(tokens) > prefix.arena.trunkLen {
		return nil, fmt.Errorf("invalid NVIDIA prefixed branch")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	h := g.model.Config.HiddenSize
	last, err := nvidia.Malloc(h)
	if err != nil {
		return nil, err
	}
	defer last.Free()
	for start := 0; start < len(tokens); start += 512 {
		end := min(start+512, len(tokens))
		dst := last
		if end != len(tokens) {
			dst = nil
		}
		if err := g.runPrefillBatchOutput(ctx, tokens[start:end], len(prefix.tokens)+start, prefix.arena, nil, dst); err != nil {
			return nil, err
		}
	}
	return g.finishSelectedDevice(last, candidates)
}

func (g *Gemma4NVIDIA) RefillPrefixed(ctx context.Context, prefix *Gemma4NVIDIAContext, suffix []int) (*Gemma4NVIDIAContext, error) {
	if g == nil || ctx == nil || prefix == nil || prefix.closed || prefix.owner != g || prefix.arena == nil || len(suffix) == 0 || len(prefix.tokens)+len(suffix) > prefix.arena.trunkLen {
		return nil, fmt.Errorf("invalid NVIDIA prefix refill")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	for start := 0; start < len(suffix); start += 512 {
		end := min(start+512, len(suffix))
		if err := g.runPrefillBatch(ctx, suffix[start:end], len(prefix.tokens)+start, prefix.arena, nil); err != nil {
			return nil, err
		}
	}
	prefix.arena.trunkUsed = len(prefix.tokens) + len(suffix)
	tokens := append(append([]int(nil), prefix.tokens...), suffix...)
	return &Gemma4NVIDIAContext{owner: g, tokens: tokens, arena: prefix.arena, borrowed: true}, nil
}

func (g *Gemma4NVIDIA) ForkPrefilled(ctx context.Context, prefix *Gemma4NVIDIAContext, suffix []int, branchCap, suffixCap int) (*Gemma4NVIDIAContext, error) {
	if g == nil || ctx == nil || prefix == nil || prefix.closed || prefix.owner != g || prefix.arena == nil || len(suffix) == 0 {
		return nil, fmt.Errorf("invalid NVIDIA prefix fork")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	a, err := prefix.arena.cloneTrunk(len(prefix.tokens)+len(suffix), branchCap, suffixCap)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			a.free()
		}
	}()
	for start := 0; start < len(suffix); start += 512 {
		end := min(start+512, len(suffix))
		if err := g.runPrefillBatch(ctx, suffix[start:end], len(prefix.tokens)+start, a, nil); err != nil {
			return nil, err
		}
	}
	tokens := append(append([]int(nil), prefix.tokens...), suffix...)
	ok = true
	return &Gemma4NVIDIAContext{owner: g, tokens: tokens, arena: a}, nil
}

type Gemma4NVIDIAContext struct {
	owner    *Gemma4NVIDIA
	tokens   []int
	arena    *gemma4NVIDIAKVArena
	closed   bool
	borrowed bool
}

func (c *Gemma4NVIDIAContext) Close() {
	if c == nil || c.closed {
		return
	}
	c.closed = true
	if c.arena != nil && !c.borrowed {
		c.arena.free()
	}
	c.arena = nil
}

func (g *Gemma4NVIDIA) runPrefillToken(ctx context.Context, token, pos int, arena *gemma4NVIDIAKVArena) error {
	m := g.model
	B, h := 1, m.Config.HiddenSize
	hiddenHost := make([]float32, h)
	if err := m.ScaledTokenEmbeddingInto(hiddenHost, token); err != nil {
		return err
	}
	hidden, err := nvidia.Malloc(h)
	if err != nil {
		return err
	}
	defer hidden.Free()
	if err = hidden.Upload(hiddenHost); err != nil {
		return err
	}
	residual, _ := nvidia.Malloc(h)
	normed, _ := nvidia.Malloc(h)
	defer residual.Free()
	defer normed.Free()
	maxQ, maxKV, maxInter := 1, 1, 1
	for l := range m.Layers {
		hd, _ := m.LayerHeadDim(l)
		maxQ = max(maxQ, m.Config.NumHeads*hd)
		maxKV = max(maxKV, gemmacfg.LayerKVHeads(m.Config, l)*hd)
		maxInter = max(maxInter, m.layerInterFor(&m.Layers[l]))
	}
	q, _ := nvidia.Malloc(maxQ)
	k, _ := nvidia.Malloc(maxKV)
	v, _ := nvidia.Malloc(maxKV)
	attn, _ := nvidia.Malloc(maxQ)
	o, _ := nvidia.Malloc(h)
	gate, _ := nvidia.Malloc(maxInter)
	up, _ := nvidia.Malloc(maxInter)
	down, _ := nvidia.Malloc(h)
	defer q.Free()
	defer k.Free()
	defer v.Free()
	defer attn.Free()
	defer o.Free()
	defer gate.Free()
	defer up.Free()
	defer down.Free()
	active, _ := nvidia.Malloc(1)
	defer active.Free()
	_ = active.UploadUint32([]uint32{0})
	for l := 0; l < m.Config.NumLayers; l++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		src := &m.Layers[l]
		gl := &g.layers[l]
		hd, _ := m.LayerHeadDim(l)
		kvHeads := gemmacfg.LayerKVHeads(m.Config, l)
		kvDim := kvHeads * hd
		inter := m.layerInterFor(src)
		if err := nvidia.CopyDtoD(residual.Ptr, hidden.Ptr, uint64(h*4)); err != nil {
			return err
		}
		if err := nvidia.IdeogramRMSNormRowsBuffer(normed, hidden, gl.inputNorm, nil, B, h, float32(m.Config.RMSNormEps), false); err != nil {
			return err
		}
		if err := gl.q.ProjectBatchToBuffer(q, normed, B); err != nil {
			return err
		}
		if src.HasKV {
			if err := gl.k.ProjectBatchToBuffer(k, normed, B); err != nil {
				return err
			}
			if gl.v == gl.k {
				if err := nvidia.CopyDtoD(v.Ptr, k.Ptr, uint64(kvDim*4)); err != nil {
					return err
				}
			} else if err := gl.v.ProjectBatchToBuffer(v, normed, B); err != nil {
				return err
			}
		}
		if gl.qNorm != nil {
			if err := nvidia.IdeogramRMSNormRowsBuffer(q, q, gl.qNorm, nil, m.Config.NumHeads, hd, float32(m.Config.RMSNormEps), false); err != nil {
				return err
			}
			if src.HasKV {
				if err := nvidia.IdeogramRMSNormRowsBuffer(k, k, gl.kNorm, nil, kvHeads, hd, float32(m.Config.RMSNormEps), false); err != nil {
					return err
				}
			}
		}
		if src.HasKV {
			if err := nvidia.IdeogramRMSNormRowsNoWeightBuffer(v, v, kvHeads, hd, float32(m.Config.RMSNormEps)); err != nil {
				return err
			}
		}
		_, rot := m.ensureGemma4RoPE(l, pos)
		rope := g.ropeSWA
		if len(m.Config.LayerTypes) <= l || m.Config.LayerTypes[l] != "sliding_attention" {
			rope = g.ropeFull
		}
		if err := nvidia.RoPEPartialRowsBuffer(q, rope, B, pos, m.Config.NumHeads, hd, rot); err != nil {
			return err
		}
		if src.HasKV {
			if err := nvidia.RoPEPartialRowsBuffer(k, rope, B, pos, kvHeads, hd, rot); err != nil {
				return err
			}
			if err := arena.appendTrunkRow(l, pos, k, v); err != nil {
				return err
			}
		}
		kvLayer := l
		if !src.HasKV {
			kvLayer = src.KVSourceLayer
		}
		start := 0
		if m.Config.SlidingWindow > 0 && len(m.Config.LayerTypes) > l && m.Config.LayerTypes[l] == "sliding_attention" && pos+1 > m.Config.SlidingWindow {
			start = pos + 1 - m.Config.SlidingWindow
		}
		if err := nvidia.IndependentBranchAttentionBuffer(attn, q, arena.trunkK[kvLayer], arena.trunkV[kvLayer], nil, nil, active, 1, 1, arena.trunkLen, 0, start, pos+1-start, m.Config.NumHeads, kvHeads, hd, attentionScale(m.Config, hd)); err != nil {
			return err
		}
		if err := gl.o.ProjectBatchToBuffer(o, attn, 1); err != nil {
			return err
		}
		if gl.preFFNNorm == nil {
			return fmt.Errorf("layer %d missing Gemma4 pre-FFN norm", l)
		}
		if err := nvidia.IdeogramRMSNormRowsBuffer(o, o, gl.postNorm, nil, 1, h, float32(m.Config.RMSNormEps), false); err != nil {
			return err
		}
		if err := nvidia.VecAddF32Buffer(residual, o, hidden, h); err != nil {
			return err
		}
		if err := nvidia.CopyDtoD(residual.Ptr, hidden.Ptr, uint64(h*4)); err != nil {
			return err
		}
		if err := nvidia.IdeogramRMSNormRowsBuffer(normed, hidden, gl.preFFNNorm, nil, 1, h, float32(m.Config.RMSNormEps), false); err != nil {
			return err
		}
		if err := gl.gate.ProjectBatchToBuffer(gate, normed, 1); err != nil {
			return err
		}
		if err := gl.up.ProjectBatchToBuffer(up, normed, 1); err != nil {
			return err
		}
		if err := nvidia.GELUTanhMulBuffer(gate, up, inter); err != nil {
			return err
		}
		if err := gl.down.ProjectBatchToBuffer(down, gate, 1); err != nil {
			return err
		}
		if gl.postFFNNorm != nil {
			if err := nvidia.IdeogramRMSNormRowsBuffer(down, down, gl.postFFNNorm, nil, 1, h, float32(m.Config.RMSNormEps), false); err != nil {
				return err
			}
		}
		if err := nvidia.VecAddF32Buffer(residual, down, hidden, h); err != nil {
			return err
		}
		if src.LayerScalar != 1 {
			if err := nvidia.VecScaleF32Buffer(hidden, hidden, h, src.LayerScalar); err != nil {
				return err
			}
		}
	}
	return nil
}

func (g *Gemma4NVIDIA) ScoreIndependentBranches(ctx context.Context, trunk MTPPromptContext, branches [][]int) (Gemma4BranchBatchResult, error) {
	if g == nil {
		return Gemma4BranchBatchResult{}, fmt.Errorf("nil NVIDIA Gemma4")
	}
	if ctx == nil {
		return Gemma4BranchBatchResult{}, fmt.Errorf("nil context")
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return Gemma4BranchBatchResult{}, fmt.Errorf("NVIDIA Gemma4 closed")
	}
	m := g.model
	if err := m.validateGemma4IndependentBranchTrunk(trunk.SeqLen, trunk.KVCacheK, trunk.KVCacheV); err != nil {
		return Gemma4BranchBatchResult{}, err
	}
	if len(branches) == 0 || len(branches) > 256 {
		return Gemma4BranchBatchResult{}, fmt.Errorf("invalid NVIDIA branch count=%d", len(branches))
	}
	maxDepth := 0
	for i, b := range branches {
		if len(b) == 0 {
			return Gemma4BranchBatchResult{}, fmt.Errorf("branch %d empty", i)
		}
		if len(b) > maxDepth {
			maxDepth = len(b)
		}
		for _, tok := range b {
			if tok < 0 || tok >= m.Config.VocabSize {
				return Gemma4BranchBatchResult{}, fmt.Errorf("branch %d token %d outside vocab", i, tok)
			}
		}
	}
	if trunk.SeqLen+maxDepth > 2048 {
		return Gemma4BranchBatchResult{}, fmt.Errorf("NVIDIA Go System One attention supports at most 2048 visible tokens")
	}
	out := Gemma4BranchBatchResult{Logits: make([][]float32, len(branches))}
	kvArena, err := newGemma4NVIDIAKVArena(m, trunk, len(branches), maxDepth)
	if err != nil {
		return Gemma4BranchBatchResult{}, err
	}
	defer kvArena.free()
	for depth := 0; depth < maxDepth; depth++ {
		if err := ctx.Err(); err != nil {
			return Gemma4BranchBatchResult{}, err
		}
		active := []int{}
		for i := range branches {
			if depth < len(branches[i]) {
				active = append(active, i)
			}
		}
		hidden, err := g.runDepth(ctx, trunk, depth, active, branches, kvArena)
		if err != nil {
			return Gemma4BranchBatchResult{}, fmt.Errorf("depth %d: %w", depth, err)
		}
		for row, branch := range active {
			if depth+1 != len(branches[branch]) {
				continue
			}
			logits, err := g.finishRow(hidden[row])
			if err != nil {
				return Gemma4BranchBatchResult{}, fmt.Errorf("branch %d finish: %w", branch, err)
			}
			out.Logits[branch] = logits
		}
	}
	return out, nil
}

func (c *Gemma4NVIDIAContext) ScoreIndependentBranches(ctx context.Context, branches [][]int) (Gemma4BranchBatchResult, error) {
	if c == nil || c.closed || c.owner == nil || c.arena == nil {
		return Gemma4BranchBatchResult{}, fmt.Errorf("NVIDIA context closed")
	}
	g := c.owner
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return Gemma4BranchBatchResult{}, fmt.Errorf("NVIDIA Gemma4 closed")
	}
	if len(branches) == 1 && len(branches[0]) > 1 {
		branch := branches[0]
		for _, tok := range branch {
			if tok < 0 || tok >= g.model.Config.VocabSize {
				return Gemma4BranchBatchResult{}, fmt.Errorf("branch token %d outside vocab", tok)
			}
		}
		final := make([]float32, g.model.Config.HiddenSize)
		if len(c.tokens)+len(branch) <= c.arena.trunkLen {
			if err := g.runPrefillBatch(ctx, branch, len(c.tokens), c.arena, final); err != nil {
				return Gemma4BranchBatchResult{}, err
			}
		} else {
			if len(branch) > c.arena.suffixCap {
				return Gemma4BranchBatchResult{}, fmt.Errorf("branch exceeds NVIDIA suffix capacity")
			}
			trunk := MTPPromptContext{Tokens: c.tokens, SeqLen: len(c.tokens)}
			for depth := range branch {
				hidden, err := g.runDepth(ctx, trunk, depth, []int{0}, [][]int{branch}, c.arena)
				if err != nil {
					return Gemma4BranchBatchResult{}, err
				}
				copy(final, hidden[0])
			}
		}
		logits, err := g.finishRow(final)
		if err != nil {
			return Gemma4BranchBatchResult{}, err
		}
		return Gemma4BranchBatchResult{Logits: [][]float32{logits}}, nil
	}
	maxDepth := 0
	for i, b := range branches {
		if len(b) == 0 || len(b) > c.arena.suffixCap {
			return Gemma4BranchBatchResult{}, fmt.Errorf("branch %d invalid depth", i)
		}
		maxDepth = max(maxDepth, len(b))
	}
	if len(branches) == 0 || len(branches) > 256 || len(branches) > c.arena.branches {
		return Gemma4BranchBatchResult{}, fmt.Errorf("invalid NVIDIA branch count=%d", len(branches))
	}
	trunk := MTPPromptContext{Tokens: c.tokens, SeqLen: len(c.tokens)}
	out := Gemma4BranchBatchResult{Logits: make([][]float32, len(branches))}
	for depth := 0; depth < maxDepth; depth++ {
		active := []int{}
		for i := range branches {
			if depth < len(branches[i]) {
				active = append(active, i)
			}
		}
		hidden, err := g.runDepth(ctx, trunk, depth, active, branches, c.arena)
		if err != nil {
			return out, err
		}
		for row, branch := range active {
			if depth+1 == len(branches[branch]) {
				out.Logits[branch], err = g.finishRow(hidden[row])
				if err != nil {
					return out, err
				}
			}
		}
	}
	return out, nil
}

func (g *Gemma4NVIDIA) runDepth(ctx context.Context, trunk MTPPromptContext, depth int, active []int, branches [][]int, kvArena *gemma4NVIDIAKVArena) ([][]float32, error) {
	m := g.model
	B, h := len(active), m.Config.HiddenSize
	hiddenHost := make([]float32, B*h)
	positions := make([]int, B)
	for row, branch := range active {
		if err := m.ScaledTokenEmbeddingInto(hiddenHost[row*h:(row+1)*h], branches[branch][depth]); err != nil {
			return nil, err
		}
		positions[row] = trunk.SeqLen + depth
	}
	hidden, err := nvidia.Malloc(B * h)
	if err != nil {
		return nil, err
	}
	defer hidden.Free()
	if err = hidden.Upload(hiddenHost); err != nil {
		return nil, err
	}
	residual, _ := nvidia.Malloc(B * h)
	normed, _ := nvidia.Malloc(B * h)
	defer residual.Free()
	defer normed.Free()
	maxQ, maxKV, maxInter := 1, 1, 1
	for l := range m.Layers {
		hd, _ := m.LayerHeadDim(l)
		if q := m.Config.NumHeads * hd; q > maxQ {
			maxQ = q
		}
		if kv := gemmacfg.LayerKVHeads(m.Config, l) * hd; kv > maxKV {
			maxKV = kv
		}
		if n := m.layerInterFor(&m.Layers[l]); n > maxInter {
			maxInter = n
		}
	}
	q, _ := nvidia.Malloc(B * maxQ)
	k, _ := nvidia.Malloc(B * maxKV)
	v, _ := nvidia.Malloc(B * maxKV)
	attn, _ := nvidia.Malloc(B * maxQ)
	o, _ := nvidia.Malloc(B * h)
	gate, _ := nvidia.Malloc(B * maxInter)
	up, _ := nvidia.Malloc(B * maxInter)
	gateUp, _ := nvidia.Malloc(B * maxInter * 2)
	down, _ := nvidia.Malloc(B * h)
	defer q.Free()
	defer k.Free()
	defer v.Free()
	defer attn.Free()
	defer o.Free()
	defer gate.Free()
	defer up.Free()
	defer gateUp.Free()
	defer down.Free()
	for l := 0; l < m.Config.NumLayers; l++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		src := &m.Layers[l]
		gl := &g.layers[l]
		hd, _ := m.LayerHeadDim(l)
		kvHeads := gemmacfg.LayerKVHeads(m.Config, l)
		kvDim := kvHeads * hd
		inter := m.layerInterFor(src)
		if err := nvidia.CopyDtoD(residual.Ptr, hidden.Ptr, uint64(B*h*4)); err != nil {
			return nil, err
		}
		if err := nvidia.IdeogramRMSNormRowsBuffer(normed, hidden, gl.inputNorm, nil, B, h, float32(m.Config.RMSNormEps), false); err != nil {
			return nil, err
		}
		if err := gl.q.ProjectBatchToBuffer(q, normed, B); err != nil {
			return nil, err
		}
		if src.HasKV {
			if err := gl.k.ProjectBatchToBuffer(k, normed, B); err != nil {
				return nil, err
			}
			if gl.v == gl.k {
				if err := nvidia.CopyDtoD(v.Ptr, k.Ptr, uint64(B*kvDim*4)); err != nil {
					return nil, err
				}
			} else if err := gl.v.ProjectBatchToBuffer(v, normed, B); err != nil {
				return nil, err
			}
		}
		if gl.qNorm != nil {
			if err := nvidia.IdeogramRMSNormRowsBuffer(q, q, gl.qNorm, nil, B*m.Config.NumHeads, hd, float32(m.Config.RMSNormEps), false); err != nil {
				return nil, err
			}
			if src.HasKV {
				if err := nvidia.IdeogramRMSNormRowsBuffer(k, k, gl.kNorm, nil, B*kvHeads, hd, float32(m.Config.RMSNormEps), false); err != nil {
					return nil, err
				}
			}
		}
		if src.HasKV {
			if err := nvidia.IdeogramRMSNormRowsNoWeightBuffer(v, v, B*kvHeads, hd, float32(m.Config.RMSNormEps)); err != nil {
				return nil, err
			}
		}
		_, rot := m.ensureGemma4RoPE(l, positions[0])
		rope := g.ropeSWA
		if len(m.Config.LayerTypes) <= l || m.Config.LayerTypes[l] != "sliding_attention" {
			rope = g.ropeFull
		}
		if rope == nil {
			return nil, fmt.Errorf("layer %d missing resident RoPE table", l)
		}
		if err := nvidia.RoPEPartialRowsBuffer(q, rope, B, positions[0], m.Config.NumHeads, hd, rot); err != nil {
			return nil, err
		}
		if src.HasKV {
			if err := nvidia.RoPEPartialRowsBuffer(k, rope, B, positions[0], kvHeads, hd, rot); err != nil {
				return nil, err
			}
			if err := kvArena.appendRows(l, positions[0], active, k, v); err != nil {
				return nil, err
			}
		}
		kvLayer := l
		if !src.HasKV {
			kvLayer = src.KVSourceLayer
		}
		seqStart := 0
		seqEnd := positions[0] + 1
		if m.Config.SlidingWindow > 0 && len(m.Config.LayerTypes) > l && m.Config.LayerTypes[l] == "sliding_attention" && seqEnd > m.Config.SlidingWindow {
			seqStart = seqEnd - m.Config.SlidingWindow
		}
		seqLen := seqEnd - seqStart
		activeBuf, err := kvArena.activeIndexBuffer(active)
		if err != nil {
			return nil, err
		}
		if err := nvidia.IndependentBranchAttentionBuffer(attn, q, kvArena.trunkK[kvLayer], kvArena.trunkV[kvLayer], kvArena.suffixK[kvLayer], kvArena.suffixV[kvLayer], activeBuf, B, kvArena.branches, kvArena.trunkUsed, kvArena.suffixCap, seqStart, seqLen, m.Config.NumHeads, kvHeads, hd, attentionScale(m.Config, hd)); err != nil {
			activeBuf.Free()
			return nil, err
		}
		activeBuf.Free()
		if err := gl.o.ProjectBatchToBuffer(o, attn, B); err != nil {
			return nil, err
		}
		if gl.preFFNNorm != nil {
			if err := nvidia.IdeogramRMSNormRowsBuffer(o, o, gl.postNorm, nil, B, h, float32(m.Config.RMSNormEps), false); err != nil {
				return nil, err
			}
			if err := nvidia.VecAddF32Buffer(residual, o, hidden, B*h); err != nil {
				return nil, err
			}
			if err := nvidia.CopyDtoD(residual.Ptr, hidden.Ptr, uint64(B*h*4)); err != nil {
				return nil, err
			}
			if err := nvidia.IdeogramRMSNormRowsBuffer(normed, hidden, gl.preFFNNorm, nil, B, h, float32(m.Config.RMSNormEps), false); err != nil {
				return nil, err
			}
		} else {
			return nil, fmt.Errorf("layer %d missing Gemma4 pre-FFN norm", l)
		}
		if gl.gateUp != nil {
			if err := gl.gateUp.ProjectBatchToBuffer(gateUp, normed, B); err != nil {
				return nil, err
			}
			if err := nvidia.GateUpGELUBuffer(gateUp, gate, B, inter); err != nil {
				return nil, err
			}
		} else {
			if err := nvidia.ProjectQ4PairToBuffers(gate, up, normed, B, gl.gate, gl.up); err != nil {
				return nil, err
			}
			if err := nvidia.GELUTanhMulBuffer(gate, up, B*inter); err != nil {
				return nil, err
			}
		}
		if err := gl.down.ProjectBatchToBuffer(down, gate, B); err != nil {
			return nil, err
		}
		if gl.postFFNNorm != nil {
			if err := nvidia.IdeogramRMSNormRowsBuffer(down, down, gl.postFFNNorm, nil, B, h, float32(m.Config.RMSNormEps), false); err != nil {
				return nil, err
			}
		}
		if err := nvidia.VecAddF32Buffer(residual, down, hidden, B*h); err != nil {
			return nil, err
		}
		if src.LayerScalar != 1 {
			if err := nvidia.VecScaleF32Buffer(hidden, hidden, B*h, src.LayerScalar); err != nil {
				return nil, err
			}
		}
	}
	outFlat := make([]float32, B*h)
	if err := hidden.Download(outFlat); err != nil {
		return nil, err
	}
	out := make([][]float32, B)
	for i := range out {
		out[i] = append([]float32(nil), outFlat[i*h:(i+1)*h]...)
	}
	return out, nil
}

func (g *Gemma4NVIDIA) FinishSelected(hidden []float32, tokens []int) ([]float32, error) {
	h := g.model.Config.HiddenSize
	if len(hidden) < h {
		return nil, fmt.Errorf("invalid selected logits")
	}
	hb, err := nvidia.Malloc(h)
	if err != nil {
		return nil, err
	}
	defer hb.Free()
	if err = hb.Upload(hidden[:h]); err != nil {
		return nil, err
	}
	return g.finishSelectedDevice(hb, tokens)
}
func (g *Gemma4NVIDIA) finishSelectedDevice(hb *nvidia.Buffer, tokens []int) ([]float32, error) {
	m := g.model
	h := m.Config.HiddenSize
	if hb == nil || hb.Size < h*4 || len(tokens) == 0 {
		return nil, fmt.Errorf("invalid selected device logits")
	}
	nb, err := nvidia.Malloc(h)
	if err != nil {
		return nil, err
	}
	defer nb.Free()
	rb, err := nvidia.MallocBytes(len(tokens) * 4)
	if err != nil {
		return nil, err
	}
	defer rb.Free()
	lb, err := nvidia.Malloc(len(tokens))
	if err != nil {
		return nil, err
	}
	defer lb.Free()
	ids := make([]uint32, len(tokens))
	for i, t := range tokens {
		if t < 0 || t >= m.Config.VocabSize {
			return nil, fmt.Errorf("candidate token outside vocab")
		}
		ids[i] = uint32(t)
	}
	if err := rb.UploadUint32(ids); err != nil {
		return nil, err
	}
	if err := nvidia.IdeogramRMSNormRowsBuffer(nb, hb, g.norm, nil, 1, h, float32(m.Config.RMSNormEps), false); err != nil {
		return nil, err
	}
	if err := g.lmHead.ProjectSelectedRows(lb, nb, rb, len(tokens)); err != nil {
		return nil, err
	}
	out := make([]float32, len(tokens))
	if err := lb.Download(out); err != nil {
		return nil, err
	}
	g.transformSelectedLogits(out, tokens)
	return out, nil
}

func (g *Gemma4NVIDIA) finishRow(hidden []float32) ([]float32, error) {
	m := g.model
	h, vocab := m.Config.HiddenSize, m.Config.VocabSize
	hb, _ := nvidia.Malloc(h)
	nb, _ := nvidia.Malloc(h)
	lb, _ := nvidia.Malloc(vocab)
	defer hb.Free()
	defer nb.Free()
	defer lb.Free()
	if err := hb.Upload(hidden); err != nil {
		return nil, err
	}
	if err := nvidia.IdeogramRMSNormRowsBuffer(nb, hb, g.norm, nil, 1, h, float32(m.Config.RMSNormEps), false); err != nil {
		return nil, err
	}
	if err := g.lmHead.ProjectBatchToBuffer(lb, nb, 1); err != nil {
		return nil, err
	}
	out := make([]float32, vocab)
	if err := lb.Download(out); err != nil {
		return nil, err
	}
	applyLlamaFinalLogitSoftcap(out, m.Config.FinalLogitSoftcapping)
	applyLlamaSuppressTokens(out, m.SuppressTokens)
	return out, nil
}
func (g *Gemma4NVIDIA) DeviceName() string {
	if g == nil {
		return ""
	}
	return nvidia.DeviceName()
}
func (g *Gemma4NVIDIA) ResidentBytes() int64 {
	if g == nil {
		return 0
	}
	seen := map[*nvidia.GPUGGUFMatrix]bool{}
	var n int64
	add := func(m *nvidia.GPUGGUFMatrix) {
		if m != nil && !seen[m] {
			n += int64(m.ResidentBytes())
			seen[m] = true
		}
	}
	for i := range g.layers {
		for _, m := range []*nvidia.GPUGGUFMatrix{g.layers[i].q, g.layers[i].k, g.layers[i].v, g.layers[i].o, g.layers[i].gate, g.layers[i].up, g.layers[i].gateUp, g.layers[i].down} {
			add(m)
		}
	}
	add(g.lmHead)
	for _, b := range []*nvidia.Buffer{g.norm, g.ropeSWA, g.ropeFull} {
		if b != nil {
			n += int64(b.Size)
		}
	}
	return n
}
