package needle

import (
	"context"
	"fmt"
	"math"
	"slices"
	"sort"
	"sync"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// DecoderOptions separates retained KV/history/prepared-weight admission from
// per-token temporary workspace. Zero cache/work limits each select 512 MiB.
// Capacity never exceeds model context or a deployed archive's KV window.
type DecoderOptions struct {
	Capacity      int
	MaxCacheBytes int64
	Execution     Options
}

// Decoder serializes calls and owns one bounded sequence. The model and prepared
// weights are immutable. A failed/cancelled Step does not publish partial state.
// K/V are float32 (including A8-dequantized values), not packed integer storage.
type Decoder struct {
	mu                 sync.Mutex
	model              *Model
	opts               Options
	capacity, position int
	bytes              int64
	params             map[string]*value
	layers             []decodeLayer
	engrams            []rowRing
	ids                []int
}
type rowRing struct {
	data        []float32
	rows, width int
}

func (r rowRing) at(pos int) []float32 {
	start := (pos % r.rows) * r.width
	return r.data[start : start+r.width]
}
func (r rowRing) put(pos int, x []float32) {
	if r.rows > 0 {
		copy(r.at(pos), x)
	}
}

type decodeLayer struct{ key, val, qraw, kraw, vraw rowRing }
type pendingLayer struct{ key, val, qraw, kraw, vraw *value }
type decodeStep struct {
	d       *Decoder
	ctx     context.Context
	layers  []pendingLayer
	engrams []*value
}

func (m *Model) NewDecoder(opts DecoderOptions) (decoder *Decoder, err error) {
	defer recoverWork(&err)
	if m == nil {
		return nil, fmt.Errorf("needle: nil model")
	}
	exec, err := m.resolveOptions(opts.Execution)
	if err != nil {
		return nil, err
	}
	if err = exec.Quant.validate(m.config.Generation); err != nil {
		return nil, err
	}
	if exec.MaxWorkBytes < 0 {
		return nil, fmt.Errorf("needle: negative workspace budget")
	}
	if exec.Quant != nil {
		q := *exec.Quant
		exec.Quant = &q
	} // caller options cannot change prepared numerics
	bound := m.config.MaxSeq
	if m.deployed && m.archiveWindow > 0 {
		bound = min(bound, m.archiveWindow)
	}
	capacity := opts.Capacity
	if capacity == 0 {
		capacity = bound
	}
	if capacity < 1 || capacity > bound {
		return nil, fmt.Errorf("needle: decoder capacity must be 1..%d", bound)
	}
	limit := opts.MaxCacheBytes
	if limit == 0 {
		limit = 512 << 20
	}
	if limit < 1 || limit > 8<<30 {
		return nil, fmt.Errorf("needle: cache budget must be 1..8 GiB")
	}
	c := m.config
	// Preflight all ring and prepared-CQ allocations before allocating any cache.
	sizes := make([][5][2]int, c.Layers)
	var retained int64 = 1024 + int64(capacity)*8 + int64(c.Layers)*256 + int64(len(c.EngramLayers))*64
	for l := range sizes {
		rows := capacity
		if c.Window > 0 && !slices.Contains(c.GlobalLayers, l) {
			rows = min(rows, c.Window)
		}
		hist := min(capacity, max(0, c.ConvTaps-1))
		sizes[l] = [5][2]int{{rows, c.KVHeads * c.QKDim}, {rows, c.KVHeads * c.VDim}, {hist, c.Heads * c.QKDim}, {hist, c.KVHeads * c.QKDim}, {hist, c.KVHeads * c.VDim}}
		for _, s := range sizes[l] {
			retained += int64(s[0]) * int64(s[1]) * 4
		}
	}
	egRows := 0
	if len(c.EngramLayers) > 0 {
		egRows = min(capacity, 3*slices.Max(c.EngramOrders))
		retained += int64(egRows) * int64(c.DModel) * 4 * int64(len(c.EngramLayers))
	}
	names := make([]string, 0)
	var largestQuant int64
	for name := range expectedShapes(c) {
		names = append(names, name)
		retained += int64(len(name)) + 256
		if !m.deployed && exec.Quant != nil && exec.Quant.WeightBits > 0 && isCQ(name) && len(m.tensors[name].Shape) >= 2 {
			n := int64(len(m.tensors[name].Data)) * 4
			retained += n
			largestQuant = max(largestQuant, n)
		}
	}
	sort.Strings(names)
	// AB broadcast/index/materialization scratch can exceed the final CQ tensor.
	// Reserve conservative construction headroom separately; CacheBytes reports
	// retained logical bytes, not transient peak or model storage/RSS.
	scratch := largestQuant*12 + int64(len(names))*512
	if retained > limit || scratch > limit-retained {
		return nil, fmt.Errorf("needle: decoder requires %d retained + %d preparation bytes, budget %d", retained, scratch, limit)
	}
	d := &Decoder{model: m, opts: exec, capacity: capacity, bytes: retained, params: make(map[string]*value, len(names)), ids: make([]int, capacity), layers: make([]decodeLayer, c.Layers), engrams: make([]rowRing, len(c.EngramLayers))}
	ring := func(rows, width int) rowRing {
		return rowRing{data: make([]float32, rows*width), rows: rows, width: width}
	}
	for l, s := range sizes {
		d.layers[l] = decodeLayer{ring(s[0][0], s[0][1]), ring(s[1][0], s[1][1]), ring(s[2][0], s[2][1]), ring(s[3][0], s[3][1]), ring(s[4][0], s[4][1])}
	}
	for i := range d.engrams {
		d.engrams[i] = ring(egRows, c.DModel)
	}
	for _, name := range names {
		src := m.tensors[name]
		v := &value{x: src.Data, r: 1, c: len(src.Data)}
		if !m.deployed && exec.Quant != nil && exec.Quant.WeightBits > 0 && isCQ(name) && len(src.Shape) >= 2 {
			e := m.execution(false, Options{MaxWorkBytes: limit - retained, Quant: exec.Quant})
			v = e.quantParam(name, v)
		}
		d.params[name] = v
	}
	return d, nil
}
func (d *Decoder) Position() int {
	if d == nil {
		return 0
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.position
}
func (d *Decoder) CacheBytes() int64 {
	if d == nil {
		return 0
	}
	return d.bytes
}

// Reset clears retained state (including previous prompt contents), retaining
// fixed cache capacity and immutable prepared weights for subsequent requests.
func (d *Decoder) Reset() {
	if d == nil {
		return
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	d.position = 0
	clear(d.ids)
	for _, l := range d.layers {
		for _, r := range []rowRing{l.key, l.val, l.qraw, l.kraw, l.vraw} {
			clear(r.data)
		}
	}
	for _, r := range d.engrams {
		clear(r.data)
	}
}
func (d *Decoder) Step(ctx context.Context, token int) (logits []float32, err error) {
	if d == nil {
		return nil, fmt.Errorf("needle: nil decoder")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	defer recoverWork(&err)
	if ctx == nil {
		return nil, fmt.Errorf("needle: nil context")
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	if token < 0 || token >= d.model.config.VocabSize {
		return nil, fmt.Errorf("needle: token out of range")
	}
	if d.position >= d.capacity {
		return nil, fmt.Errorf("needle: decoder capacity exhausted")
	}
	e := d.model.execution(false, d.opts)
	e.t.arena = &inferenceArena{limit: e.t.limit}
	e.parameterViews = d.params
	e.t.reserve(int64(len(d.layers))*96 + int64(len(d.engrams))*8)
	step := &decodeStep{d: d, ctx: ctx, layers: make([]pendingLayer, len(d.layers)), engrams: make([]*value, len(d.engrams))}
	e.decode = step
	out := e.forward([]int{token})
	for _, v := range out.x {
		if !finite(v) {
			return nil, fmt.Errorf("needle: nonfinite cached logits")
		}
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	// Return only the logits, not a view retaining an arena block.
	e.t.reserve(int64(len(out.x)) * 4)
	logits = append([]float32(nil), out.x...)
	// Every possible fallible operation is above. Commit has no allocations.
	pos := d.position
	for l, p := range step.layers {
		cache := d.layers[l]
		cache.key.put(pos, p.key.x)
		cache.val.put(pos, p.val.x)
		if p.qraw != nil {
			cache.qraw.put(pos, p.qraw.x)
			cache.kraw.put(pos, p.kraw.x)
			cache.vraw.put(pos, p.vraw.x)
		}
	}
	for i, v := range step.engrams {
		d.engrams[i].put(pos, v.x)
	}
	d.ids[pos] = token
	d.position++
	return logits, nil
}
func (e *execution) cachedConv(current, taps *value, history rowRing, dilation, window int) *value {
	out := e.t.alloc(1, current.c)
	pos := e.decode.d.position
	for j := 0; j < taps.r; j++ {
		offset := j * dilation
		if offset > pos || (window > 0 && offset >= window) {
			continue
		}
		x := current.x
		if j > 0 {
			x = history.at(pos - offset)
		}
		weights := taps.x[j*current.c : (j+1)*current.c]
		for i, v := range x {
			out.x[i] += weights[i] * v
		}
	}
	return out
}
func (e *execution) cachedAttention(x *value, l, window int) *value {
	d, t, c := e.decode.d, e.t, e.m.config
	pos := d.position
	cache := d.layers[l]
	p := "stack/layers/block/self_attn/"
	x = e.aq(x)
	q := e.linear(x, p+"q_proj/kernel", l)
	k := e.linear(x, p+"k_proj/kernel", l)
	v := e.linear(x, p+"v_proj/kernel", l)
	pending := pendingLayer{}
	if c.ConvTaps > 0 {
		pending.qraw, pending.kraw, pending.vraw = q, k, v
		q = e.cachedConv(q, e.param(p+"q_taps", l), cache.qraw, 1, window)
		k = e.cachedConv(k, e.param(p+"k_taps", l), cache.kraw, 1, window)
		v = e.cachedConv(v, e.param(p+"v_taps", l), cache.vraw, 1, window)
	}
	qs, ks := e.param(p+"q_norm/scale", l), e.param(p+"k_norm/scale", l)
	kvals, vvals := t.alloc(1, c.KVHeads*c.QKDim), t.alloc(1, c.KVHeads*c.VDim)
	for h := 0; h < c.KVHeads; h++ {
		kk := e.kvq(t.ropeAt(t.norm(t.cols(k, h*c.QKDim, c.QKDim), ks), c.RopeTheta, pos))
		vv := e.kvq(t.cols(v, h*c.VDim, c.VDim))
		copy(kvals.x[h*c.QKDim:], kk.x)
		copy(vvals.x[h*c.VDim:], vv.x)
	}
	pending.key, pending.val = kvals, vvals
	e.decode.layers[l] = pending
	count := min(pos+1, cache.key.rows)
	start := pos - count + 1
	heads := make([]*value, c.Heads)
	scale := float32(1 / math.Sqrt(float64(c.QKDim)))
	for h := 0; h < c.Heads; h++ {
		kh := h / (c.Heads / c.KVHeads)
		qh := e.aq(t.ropeAt(t.norm(t.cols(q, h*c.QKDim, c.QKDim), qs), c.RopeTheta, pos))
		scores := t.alloc(1, count)
		for j := 0; j < count; j++ {
			absolute := start + j
			row := kvals.x
			if absolute != pos {
				row = cache.key.at(absolute)
			}
			scores.x[j] = simd.Sdot(qh.x, row[kh*c.QKDim:(kh+1)*c.QKDim]) * scale
		}
		probs := t.softmax(scores, false, 0)
		o := t.alloc(1, c.VDim)
		for j, p := range probs.x {
			absolute := start + j
			row := vvals.x
			if absolute != pos {
				row = cache.val.at(absolute)
			}
			simd.Saxpy(p, row[kh*c.VDim:(kh+1)*c.VDim], o.x)
		}
		heads[h] = o
	}
	joined := t.concat(heads)
	gate := t.unary(e.linear(x, p+"gate_proj/kernel", l), "sigmoid")
	return e.linear(e.aq(t.mul(joined, gate)), p+"out_proj/kernel", l)
}
func (e *execution) cachedEngrams(token int) ([]*value, []*value) {
	c, t, d := e.m.config, e.t, e.decode.d
	keys, values := make([]*value, len(c.EngramLayers)), make([]*value, len(c.EngramLayers))
	if len(keys) == 0 {
		return keys, values
	}
	pos := d.position
	tables := len(c.EngramOrders) * c.EngramHeads
	sub := c.DModel / tables
	stride := c.EngramSeedHeads
	if stride == 0 {
		stride = c.EngramHeads
	}
	indices := t.ints(c.DModel)
	for oi, order := range c.EngramOrders {
		for h := 0; h < c.EngramHeads; h++ {
			acc := uint32(0x9e3779b9) * uint32(oi*stride+h+1)
			for j := 0; j < order; j++ {
				id := uint32(0)
				if j == 0 {
					id = uint32(token)
				} else if pos >= j {
					id = uint32(d.ids[pos-j])
				}
				acc = (acc ^ id) * 0x01000193
			}
			acc ^= acc >> 15
			table := oi*c.EngramHeads + h
			slot := int(acc % uint32(c.EngramSlots))
			for j := 0; j < sub; j++ {
				idx := table*sub + j
				indices[idx] = -1
				if pos >= order-1 {
					indices[idx] = (table*c.EngramSlots+slot)*sub + j
				}
			}
		}
	}
	dilation := slices.Max(c.EngramOrders)
	for site := range keys {
		p := fmt.Sprintf("engrams_%d/", site)
		emb := e.param(p+"embedding", -1)
		fetched := e.aq(t.gather(emb, 1, c.DModel, indices))
		keys[site] = e.linear(fetched, p+"key_proj/kernel", -1)
		v := e.linear(fetched, p+"value_proj/kernel", -1)
		e.decode.engrams[site] = v
		values[site] = e.cachedConv(v, e.param(p+"taps", -1), d.engrams[site], dilation, 0)
	}
	return keys, values
}
