package needle

import (
	"fmt"
	"math"
	"slices"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

// Options bounds per-call tape allocations. Zero selects 512 MiB. Quant is an
// explicit dequantized/STE reference mode, not packed or KV-cached decoding.
type Options struct {
	MaxWorkBytes int64
	Quant        *Quantization
	Packed       bool // opt-in direct CQ projections for an archive; dense reference remains default
}
type execution struct {
	t              *tape
	m              *Model
	p              map[string]*value
	qp             map[string]*value
	q              *Quantization
	keep           []bool // nil for ordinary LM forward; heads mask padding keys.
	cells          []*value
	parameterViews map[string]*value // immutable inference-only prepared tensors
	decode         *decodeStep
	packedEnabled  bool
}

func (m *Model) execution(train bool, opts Options) *execution {
	limit := opts.MaxWorkBytes
	if limit == 0 {
		limit = 512 << 20
	}
	return &execution{t: &tape{train: train, limit: limit}, m: m, p: map[string]*value{}, qp: map[string]*value{}, q: opts.Quant, packedEnabled: opts.Packed}
}
func (e *execution) param(name string, layer int) *value {
	if e.parameterViews != nil {
		p, ok := e.parameterViews[name]
		if !ok {
			panic("needle: missing prepared parameter " + name)
		}
		shape := e.m.tensors[name].Shape
		start := 0
		if layer >= 0 {
			start = layer * len(p.x) / shape[0]
			shape = shape[1:]
		}
		r, c := 1, 1
		if len(shape) > 0 {
			c = shape[len(shape)-1]
			for _, d := range shape[:len(shape)-1] {
				r *= d
			}
		}
		return e.t.view(p.x[start:start+r*c:start+r*c], r, c)
	}
	p, ok := e.p[name]
	if !ok {
		tensor := e.m.tensors[name]
		p = e.t.leaf(tensor.Data)
		e.p[name] = p
	}
	quantized, ok := e.qp[name]
	if !ok {
		quantized = e.quantParam(name, p)
		e.qp[name] = quantized
	}
	p = quantized
	shape := e.m.tensors[name].Shape
	start := 0
	if layer >= 0 {
		size := len(p.x) / shape[0]
		start = layer * size
		shape = shape[1:]
	}
	r, c := 1, 1
	if len(shape) > 0 {
		c = shape[len(shape)-1]
		for _, d := range shape[:len(shape)-1] {
			r *= d
		}
	}
	return e.t.slice(p, start, r, c)
}
func (e *execution) bp(key string, l int) *value { return e.param("stack/layers/block/"+key, l) }
func (e *execution) linear(x *value, key string, l int) *value {
	if e.packedEnabled {
		if p := e.m.packed[packedKey{key, l}]; p != nil {
			return e.packedLinear(x, p)
		}
	}
	return e.t.mm(x, e.param(key, l), false)
}
func (e *execution) norm(x *value, key string, l int) *value { return e.t.norm(x, e.param(key, l)) }
func (t *tape) rope(a *value, theta float64) *value          { return t.ropeAt(a, theta, 0) }
func (t *tape) ropeAt(a *value, theta float64, position int) *value {
	o := t.alloc(a.r, a.c)
	half := a.c / 2
	for r := 0; r < a.r; r++ {
		for j := 0; j < half; j++ {
			angle := float64(r+position) / math.Pow(theta, float64(2*j)/float64(a.c))
			co, si := float32(math.Cos(angle)), float32(math.Sin(angle))
			i, k := r*a.c+j, r*a.c+j+half
			o.x[i] = a.x[i]*co - a.x[k]*si
			o.x[k] = a.x[k]*co + a.x[i]*si
		}
	}
	if t.train {
		t.record(func() {
			for r := 0; r < a.r; r++ {
				for j := 0; j < half; j++ {
					angle := float64(r+position) / math.Pow(theta, float64(2*j)/float64(a.c))
					co, si := float32(math.Cos(angle)), float32(math.Sin(angle))
					i, k := r*a.c+j, r*a.c+j+half
					a.g[i] += o.g[i]*co + o.g[k]*si
					a.g[k] += -o.g[i]*si + o.g[k]*co
				}
			}
		})
	}
	return o
}
func (e *execution) attention(x *value, l, window int) *value {
	if e.decode != nil {
		return e.cachedAttention(x, l, window)
	}
	t, c := e.t, e.m.config
	p := "stack/layers/block/self_attn/"
	x = e.aq(x)
	q := e.linear(x, p+"q_proj/kernel", l)
	k := e.linear(x, p+"k_proj/kernel", l)
	v := e.linear(x, p+"v_proj/kernel", l)
	if c.ConvTaps > 0 {
		q = e.conv(q, e.param(p+"q_taps", l), 1, window, false)
		k = e.conv(k, e.param(p+"k_taps", l), 1, window, false)
		v = e.conv(v, e.param(p+"v_taps", l), 1, window, false)
	}
	qs, ks := e.param(p+"q_norm/scale", l), e.param(p+"k_norm/scale", l)
	out := make([]*value, c.Heads)
	for h := 0; h < c.Heads; h++ {
		kh := h / (c.Heads / c.KVHeads)
		qh := t.rope(t.norm(t.cols(q, h*c.QKDim, c.QKDim), qs), c.RopeTheta)
		kk := t.rope(t.norm(t.cols(k, kh*c.QKDim, c.QKDim), ks), c.RopeTheta)
		vv := t.cols(v, kh*c.VDim, c.VDim)
		qh = e.aq(qh)
		kk = e.kvq(kk)
		vv = e.kvq(vv)
		prob := e.attentionSoftmax(t.scale(t.mm(qh, kk, true), float32(1/math.Sqrt(float64(c.QKDim)))), window)
		out[h] = t.mm(prob, vv, false)
	}
	joined := t.concat(out)
	gate := t.unary(e.linear(x, p+"gate_proj/kernel", l), "sigmoid")
	return e.linear(e.aq(t.mul(joined, gate)), p+"out_proj/kernel", l)
}
func (t *tape) permute(a *value, p []int) *value {
	if !t.train {
		o := t.alloc(a.r, a.c)
		for row := 0; row < a.r; row++ {
			for col, src := range p {
				o.x[row*a.c+col] = a.x[row*a.c+src]
			}
		}
		return o
	}
	idx := t.ints(len(a.x))
	for r := 0; r < a.r; r++ {
		for j, v := range p {
			idx[r*a.c+j] = r*a.c + v
		}
	}
	return t.gather(a, a.r, a.c, idx)
}
func (t *tape) kron(z, a, b *value) *value {
	ba, bb := a.r, b.r
	if !t.train {
		za := t.alloc(z.r*bb, ba)
		for token := 0; token < z.r; token++ {
			for i := 0; i < ba; i++ {
				for j := 0; j < bb; j++ {
					za.x[(token*bb+j)*ba+i] = z.x[token*z.c+i*bb+j]
				}
			}
		}
		za = t.mm(za, a, false)
		zb := t.alloc(z.r*ba, bb)
		for token := 0; token < z.r; token++ {
			for i := 0; i < ba; i++ {
				for j := 0; j < bb; j++ {
					zb.x[(token*ba+i)*bb+j] = za.x[(token*bb+j)*ba+i]
				}
			}
		}
		return t.slice(t.mm(zb, b, false), 0, z.r, z.c)
	}
	idx := t.ints(len(z.x))
	for token := 0; token < z.r; token++ {
		for i := 0; i < ba; i++ {
			for j := 0; j < bb; j++ {
				idx[(token*bb+j)*ba+i] = token*z.c + i*bb + j
			}
		}
	}
	za := t.gather(z, z.r*bb, ba, idx)
	za = t.mm(za, a, false)
	for token := 0; token < z.r; token++ {
		for i := 0; i < ba; i++ {
			for j := 0; j < bb; j++ {
				idx[(token*ba+i)*bb+j] = (token*bb+j)*ba + i
			}
		}
	}
	zb := t.gather(za, z.r*ba, bb, idx)
	zb = t.mm(zb, b, false)
	return t.slice(zb, 0, z.r, z.c)
}
func (t *tape) walsh(z *value) *value {
	o := t.alloc(z.r, z.c)
	copy(o.x, z.x)
	apply := func(data []float32) {
		scale := float32(1 / math.Sqrt(float64(z.c)))
		for r := 0; r < z.r; r++ {
			row := data[r*z.c : (r+1)*z.c]
			for step := 1; step < z.c; step *= 2 {
				for start := 0; start < z.c; start += 2 * step {
					for j := 0; j < step; j++ {
						a, b := row[start+j], row[start+j+step]
						row[start+j] = a + b
						row[start+j+step] = a - b
					}
				}
			}
			for i := range row {
				row[i] *= scale
			}
		}
	}
	apply(o.x)
	if t.train {
		t.record(func() {
			g := append([]float32(nil), o.g...)
			apply(g)
			for i, v := range g {
				z.g[i] += v
			}
		})
	}
	return o
}
func (e *execution) hadamard(x *value, l int) *value {
	t, c := e.t, e.m.config
	n := padded(c.DModel)
	var z *value
	if !t.train {
		z = t.alloc(x.r, n)
		for r := 0; r < x.r; r++ {
			copy(z.x[r*n:], x.x[r*x.c:(r+1)*x.c])
		}
	} else {
		idx := t.ints(x.r * n)
		for r := 0; r < x.r; r++ {
			for j := 0; j < n; j++ {
				idx[r*n+j] = -1
				if j < x.c {
					idx[r*n+j] = r*x.c + j
				}
			}
		}
		z = t.gather(x, x.r, n, idx)
	}
	p := "hadamard_mlp/"
	mul := func(a *value, key string) *value { return t.mul(a, t.broadcast(e.bp(p+key, l), a.r)) }
	if c.Generation == 2 {
		z = t.walsh(mul(z, "d1"))
		z = t.walsh(t.unary(mul(z, "d2"), "silu"))
		return t.cols(mul(z, "d3"), 0, c.DModel)
	}
	cond := t.mm(t.softmax(t.mm(x, e.bp(p+"cond_v", l), false), false, 0), e.bp(p+"cond_u", l), false)
	cond = t.add(cond, t.constant(cond.r, cond.c, 1))
	kron := func(z *value, j int) *value {
		return t.kron(z, e.bp(fmt.Sprintf("%sw%da", p, j), l), e.bp(fmt.Sprintf("%sw%db", p, j), l))
	}
	z = t.permute(kron(mul(z, "d1"), 1), e.m.p1)
	z = t.unary(t.add(t.mul(mul(z, "d2"), cond), t.broadcast(e.bp(p+"b2", l), x.r)), "silu")
	z = t.permute(kron(z, 2), e.m.p2)
	z = kron(mul(z, "d3"), 3)
	return t.cols(mul(z, "d4"), 0, c.DModel)
}
func (e *execution) engrams(ids []int) ([]*value, []*value) {
	if e.decode != nil {
		return e.cachedEngrams(ids[0])
	}
	c, t := e.m.config, e.t
	keys, values := make([]*value, len(c.EngramLayers)), make([]*value, len(c.EngramLayers))
	if len(keys) == 0 {
		return keys, values
	}
	tables := len(c.EngramOrders) * c.EngramHeads
	sub := c.DModel / tables
	indices := t.ints(len(ids) * c.DModel)
	stride := c.EngramSeedHeads
	if stride == 0 {
		stride = c.EngramHeads
	}
	for token := range ids {
		for oi, order := range c.EngramOrders {
			for h := 0; h < c.EngramHeads; h++ {
				acc := uint32(0x9e3779b9) * uint32(oi*stride+h+1)
				for j := 0; j < order; j++ {
					id := uint32(0)
					if token >= j {
						id = uint32(ids[token-j])
					}
					acc = (acc ^ id) * 0x01000193
				}
				acc ^= acc >> 15
				table := oi*c.EngramHeads + h
				slot := int(acc % uint32(c.EngramSlots))
				for j := 0; j < sub; j++ {
					i := (token*tables+table)*sub + j
					indices[i] = -1
					if token >= order-1 && (e.keep == nil || e.keep[token-order+1]) {
						indices[i] = (table*c.EngramSlots+slot)*sub + j
					}
				}
			}
		}
	}
	dilation := slices.Max(c.EngramOrders)
	for site := range keys {
		p := fmt.Sprintf("engrams_%d/", site)
		emb := e.param(p+"embedding", -1)
		fetched := e.aq(t.gather(emb, len(ids), c.DModel, indices))
		keys[site] = e.linear(fetched, p+"key_proj/kernel", -1)
		v := e.linear(fetched, p+"value_proj/kernel", -1)
		values[site] = e.conv(v, e.param(p+"taps", -1), dilation, 0, true)
	}
	return keys, values
}
func (e *execution) block(u *value, l int, ek, ev *value) *value {
	t, c := e.t, e.m.config
	x := u
	if ek != nil {
		a, b := t.rms(x), t.rms(ek)
		product := t.mul(a, b)
		alpha := t.alloc(x.r, 1)
		for r := 0; r < x.r; r++ {
			for _, v := range product.x[r*x.c : (r+1)*x.c] {
				alpha.x[r] += v
			}
			alpha.x[r] /= float32(math.Sqrt(float64(x.c)))
		}
		if t.train {
			t.record(func() {
				for r := 0; r < x.r; r++ {
					for j := 0; j < x.c; j++ {
						product.g[r*x.c+j] += alpha.g[r] / float32(math.Sqrt(float64(x.c)))
					}
				}
			})
		}
		gate := t.unary(alpha, "sigmoid")
		idx := t.ints(len(x.x))
		for i := range idx {
			idx[i] = i / x.c
		}
		x = t.add(x, t.mul(t.gather(gate, x.r, x.c, idx), ev))
	}
	skip := x
	norm := t.norm(x, e.bp("ZCRMSNorm_0/scale", l))
	window := c.Window
	if slices.Contains(c.GlobalLayers, l) {
		window = 0
	}
	x = e.attention(norm, l, window)
	x = t.norm(x, e.bp("post_attn_norm/scale", l))
	gate := t.unary(e.bp("attn_gate", l), "sigmoid")
	idx := t.ints(len(x.x))
	x = t.add(skip, t.mul(x, t.gather(gate, x.r, x.c, idx)))
	return t.add(x, e.hadamard(t.norm(x, e.bp("pre_hada_norm/scale", l)), l))
}
func (e *execution) forward(ids []int) *value {
	return e.outputProjection(e.trunk(ids, false))
}
func (e *execution) trunk(ids []int, collect bool) *value {
	c, t := e.m.config, e.t
	embedding := e.param("embedding/embedding", -1)
	indices := t.ints(len(ids) * c.DModel)
	for i, id := range ids {
		for j := 0; j < c.DModel; j++ {
			indices[i*c.DModel+j] = id*c.DModel + j
		}
	}
	x := t.scale(t.gather(embedding, len(ids), c.DModel, indices), float32(math.Sqrt(float64(c.DModel))))
	if collect {
		e.cells = append(e.cells, x)
	}
	lanes := make([]*value, c.Lanes)
	for i := range lanes {
		lanes[i] = x
	}
	stream := t.concat(lanes)
	ek, ev := e.engrams(ids)
	for l := 0; l < c.Layers; l++ {
		if e.decode != nil {
			if err := e.decode.ctx.Err(); err != nil {
				panic(workLimit{err})
			}
		}
		nx := e.aq(t.rms(stream))
		hc := func(kind string, cols int) *value {
			a := e.param("stack/mhc_a_"+kind, l)
			bias := e.param("stack/mhc_b_"+kind, l)
			prod := e.linear(nx, "stack/mhc_phi_"+kind, l)
			idx := t.ints(len(prod.x))
			scaled := t.mul(prod, t.gather(a, prod.r, prod.c, idx))
			return t.add(scaled, t.broadcast(t.slice(bias, 0, 1, cols), len(ids)))
		}
		pre, post := hc("pre", c.Lanes), hc("post", c.Lanes)
		offPre, offPost := t.constant(len(ids), c.Lanes, -4), t.constant(len(ids), c.Lanes, -4)
		for r := range ids {
			offPre.x[r*c.Lanes+l%c.Lanes] = 4
			offPost.x[r*c.Lanes+l%c.Lanes] = 0
		}
		pre = t.unary(t.add(pre, offPre), "sigmoid")
		post = t.scale(t.unary(t.add(post, offPost), "sigmoid"), 2)
		u := t.constant(len(ids), c.DModel, 0)
		for lane := 0; lane < c.Lanes; lane++ {
			ls := t.cols(stream, lane*c.DModel, c.DModel)
			g := t.cols(pre, lane, 1)
			idx := t.ints(len(ids) * c.DModel)
			for i := range idx {
				idx[i] = i / c.DModel
			}
			u = t.add(u, t.mul(ls, t.gather(g, len(ids), c.DModel, idx)))
		}
		var k, v *value
		for site, layer := range c.EngramLayers {
			if layer == l {
				k, v = ek[site], ev[site]
			}
		}
		y := t.add(e.block(u, l, k, v), t.scale(u, -1))
		res := hc("res", c.Lanes*c.Lanes)
		tokens := make([]*value, len(ids))
		for token := range ids {
			mat := t.slice(res, token*c.Lanes*c.Lanes, c.Lanes, c.Lanes)
			mat = t.sinkhorn(mat)
			old := t.slice(stream, token*c.Lanes*c.DModel, c.Lanes, c.DModel)
			mixed := t.mm(mat, old, false)
			g := t.slice(post, token*c.Lanes, c.Lanes, 1)
			yy := t.rows(y, token, 1)
			delta := t.mm(g, yy, false)
			tokens[token] = t.slice(t.add(mixed, delta), 0, 1, c.Lanes*c.DModel)
		}
		// Concatenate token rows without changing the stream's token-major layout.
		joined := t.concat(tokens)
		stream = t.slice(joined, 0, len(ids), c.Lanes*c.DModel)
		if collect {
			e.cells = append(e.cells, e.meanLanes(stream))
		}
	}
	x = e.meanLanes(stream)
	if collect {
		return x
	} // Upstream heads collect cells before final_norm.
	return e.norm(x, "stack/final_norm/scale", -1)
}
func (m *Model) validateTokens(ids []int) error {
	if m == nil {
		return fmt.Errorf("needle: nil model")
	}
	if m.deployed && m.archiveWindow > 0 && len(ids) > m.archiveWindow {
		return fmt.Errorf("needle: prefix exceeds archive KV window; streaming window eviction not implemented")
	}
	if len(ids) == 0 || len(ids) > m.config.MaxSeq {
		return fmt.Errorf("needle: token count must be 1..%d", m.config.MaxSeq)
	}
	for _, id := range ids {
		if id < 0 || id >= m.config.VocabSize {
			return fmt.Errorf("needle: token %d out of range", id)
		}
	}
	return nil
}
func recoverWork(err *error) {
	if v := recover(); v != nil {
		if e, ok := v.(workLimit); ok {
			*err = e.error
		} else {
			panic(v)
		}
	}
}
func (m *Model) resolveOptions(opts Options) (Options, error) {
	if opts.Packed && (!m.deployed || len(m.packed) == 0) {
		return opts, fmt.Errorf("needle: packed projections require original .cact weights")
	}
	if m.deployed {
		if opts.Quant != nil && (opts.Quant.WeightBits != 0 || opts.Quant.ActivationBits != 8 || opts.Quant.KVBits != 8) {
			return opts, fmt.Errorf("needle: archive already quantized; numerics fixed to A8/KV8")
		}
		opts.Quant = &Quantization{ActivationBits: 8, KVBits: 8}
	}
	return opts, nil
}
func (m *Model) Forward(ids []int, opts Options) (logits []float32, err error) {
	defer recoverWork(&err)
	if err = m.validateTokens(ids); err != nil {
		return nil, err
	}
	if err = opts.Quant.validate(m.config.Generation); err != nil {
		return nil, err
	}
	if opts, err = m.resolveOptions(opts); err != nil {
		return nil, err
	}
	e := m.execution(false, opts)
	out := e.forward(ids)
	return out.x, nil
}

// LossGrad computes teacher-forced next-token CE. mask[i] weights the target
// ids[i+1]; nil includes every next-token position. Gradients are native FP32;
// Quant optionally applies Needle3 CQ/A8/KV8 straight-through numerics.
func (m *Model) LossGrad(ids []int, mask []float32, opts Options) (loss float64, grad map[string]checkpoint.Tensor, err error) {
	defer recoverWork(&err)
	if err = m.validateTokens(ids); err != nil {
		return 0, nil, err
	}
	if m.deployed || opts.Packed {
		return 0, nil, fmt.Errorf("needle: train from source safetensors, not deployed/packed weights")
	}
	if len(ids) < 2 {
		return 0, nil, fmt.Errorf("needle: training requires at least two tokens")
	}
	if mask != nil && len(mask) != len(ids)-1 {
		return 0, nil, fmt.Errorf("needle: invalid loss mask length")
	}
	var total float64
	for i := 0; i < len(ids)-1; i++ {
		v := float32(1)
		if mask != nil {
			v = mask[i]
		}
		if v < 0 || math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, nil, fmt.Errorf("needle: invalid loss mask")
		}
		total += float64(v)
		if ids[i+1] >= m.config.OutVocab {
			return 0, nil, fmt.Errorf("needle: target outside output vocabulary")
		}
	}
	if total == 0 {
		return 0, nil, fmt.Errorf("needle: loss mask has no supervised targets")
	}
	if err = opts.Quant.validate(m.config.Generation); err != nil {
		return 0, nil, err
	}
	e := m.execution(true, opts)
	out := e.forward(ids)
	for row := 0; row < len(ids)-1; row++ {
		weight := float64(1)
		if mask != nil {
			weight = float64(mask[row])
		}
		off := row * out.c
		mx := out.x[off]
		for _, v := range out.x[off : off+out.c] {
			if v > mx {
				mx = v
			}
		}
		var sum float64
		for _, v := range out.x[off : off+out.c] {
			sum += math.Exp(float64(v - mx))
		}
		loss += weight * (math.Log(sum) + float64(mx-out.x[off+ids[row+1]])) / total
		for j := 0; j < out.c; j++ {
			out.g[off+j] = float32(weight / total * math.Exp(float64(out.x[off+j]-mx)) / sum)
		}
		out.g[off+ids[row+1]] -= float32(weight / total)
	}
	e.t.back()
	grad = make(map[string]checkpoint.Tensor, len(e.p))
	for name, p := range e.p {
		grad[name] = checkpoint.Tensor{Shape: append([]int{}, m.tensors[name].Shape...), Data: p.g}
	}
	return loss, grad, nil
}
