package needle

import (
	"fmt"
	"math"
	"slices"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func (e *execution) meanLanes(stream *value) *value {
	c, t := e.m.config, e.t
	x := t.constant(stream.r, c.DModel, 0)
	for lane := 0; lane < c.Lanes; lane++ {
		x = t.add(x, t.scale(t.cols(stream, lane*c.DModel, c.DModel), 1/float32(c.Lanes)))
	}
	return x
}
func (e *execution) conv(a, taps *value, dilation, window int, maskCurrent bool) *value {
	if e.keep == nil {
		return e.t.conv(a, taps, dilation, window)
	}
	t := e.t
	o := t.constant(a.r, a.c, 0)
	for j := 0; j < taps.r; j++ {
		offset := j * dilation
		idx := t.ints(len(a.x))
		for r := 0; r < a.r; r++ {
			for c := 0; c < a.c; c++ {
				idx[r*a.c+c] = -1
				if r >= offset && (window == 0 || offset < window) && ((!maskCurrent && offset == 0) || e.keep[r-offset]) {
					idx[r*a.c+c] = (r-offset)*a.c + c
				}
			}
		}
		o = t.add(o, t.mul(t.gather(a, a.r, a.c, idx), t.broadcast(t.rows(taps, j, 1), a.r)))
	}
	return o
}
func (e *execution) attentionSoftmax(a *value, window int) *value {
	if e.keep == nil {
		return e.t.softmax(a, true, window)
	}
	t := e.t
	masked := t.alloc(a.r, a.c)
	copy(masked.x, a.x)
	for r := 0; r < a.r; r++ {
		for col := 0; col < a.c; col++ {
			if col > r || (window > 0 && r-col >= window) || !e.keep[col] {
				masked.x[r*a.c+col] = -math.MaxFloat32
			}
		}
	}
	if t.train {
		t.record(func() {
			for r := 0; r < a.r; r++ {
				for col := 0; col < a.c; col++ {
					if col <= r && (window == 0 || r-col < window) && e.keep[col] {
						a.g[r*a.c+col] += masked.g[r*a.c+col]
					}
				}
			}
		})
	}
	return t.softmax(masked, false, 0)
}

// HeadKind names the Needle3 auxiliary heads. Confidence and Router return raw
// logits, not calibrated probabilities or claims of correctness after tuning.
type HeadKind string

const (
	Embedding   HeadKind = "embedding"
	Contrastive HeadKind = "contrastive"
	Confidence  HeadKind = "confidence"
	Router      HeadKind = "router"
)

func (m *Model) headGeometry(kind HeadKind) (k, q, out int, err error) {
	if m.config.Generation == 2 {
		return m.headGeometryV2(kind)
	}
	if m.config.Generation != 3 {
		return 0, 0, 0, fmt.Errorf("needle: probe-head API requires Needle3")
	}
	if kind != Embedding && kind != Confidence && kind != Router {
		return 0, 0, 0, fmt.Errorf("needle: unknown head %q", kind)
	}
	c := m.config
	wantK, wantQ, wantDim := c.EmbeddingProbes, c.EmbeddingQueries, c.EmbeddingDim
	if kind == Confidence {
		wantK, wantQ, wantDim = c.ConfidenceProbes, c.ConfidenceQueries, 1
	}
	if kind == Router {
		wantK, wantQ, wantDim = c.RouterProbes, c.RouterQueries, 3
	}
	prefix := string(kind) + "_head/"
	p, ok := m.tensors[prefix+"probes"]
	if !ok {
		return 0, 0, 0, fmt.Errorf("needle: %s head weights not loaded", kind)
	}
	l, d := m.config.Layers+1, m.config.DModel
	if len(p.Shape) != 3 || p.Shape[0] != l || p.Shape[2] != d || p.Shape[1] < 1 || p.Shape[1] > 64 {
		return 0, 0, 0, fmt.Errorf("needle: invalid %s probes", kind)
	}
	k = p.Shape[1]
	query, ok := m.tensors[prefix+"query"]
	if !ok || len(query.Shape) != 2 || query.Shape[0] < 1 || query.Shape[0] > 64 || query.Shape[1] != d {
		return 0, 0, 0, fmt.Errorf("needle: invalid %s query", kind)
	}
	q = query.Shape[0]
	proj, ok := m.tensors[prefix+"proj/kernel"]
	if !ok || len(proj.Shape) != 2 || proj.Shape[0] != q*d || proj.Shape[1] < 1 || proj.Shape[1] > 4096 {
		return 0, 0, 0, fmt.Errorf("needle: invalid %s projection", kind)
	}
	out = proj.Shape[1]
	if k != wantK || q != wantQ || out != wantDim {
		return 0, 0, 0, fmt.Errorf("needle: %s head disagrees with configured geometry", kind)
	}
	if (kind == Confidence && out != 1) || (kind == Router && out != 3) {
		return 0, 0, 0, fmt.Errorf("needle: invalid %s output size", kind)
	}
	for key, shape := range map[string][]int{"gain": {l, k}, "row_bias": {q, l, k}} {
		if got, ok := m.tensors[prefix+key]; !ok || !slices.Equal(got.Shape, shape) {
			return 0, 0, 0, fmt.Errorf("needle: invalid %s %s", kind, key)
		}
	}
	if kind != Embedding {
		if got, ok := m.tensors[prefix+"proj/bias"]; !ok || !slices.Equal(got.Shape, []int{out}) {
			return 0, 0, 0, fmt.Errorf("needle: invalid %s projection bias", kind)
		}
	}
	return k, q, out, nil
}
func (e *execution) headWeight(name string) *value {
	// Upstream first cq_ste_params quantizes kernels, then head_weight applies
	// its own W4 CQ; probes/query only get the latter. Preserve that order.
	v := e.param(name, -1)
	if e.q == nil || e.m.deployed {
		return v
	}
	o := e.t.alloc(v.r, v.c)
	cqValues(o.x, v.x, e.m.tensors[name].Shape, cqSecondLast(name), 4)
	for _, f := range o.x {
		if !finite(f) {
			panic(workLimit{fmt.Errorf("needle: head CQ overflow")})
		}
	}
	if e.t.train {
		e.t.record(func() { simd.Saxpy(1, o.g, v.g) })
	}
	return o
}

// Head runs upstream's padding-aware pooling over embedding and per-layer cells.
// It never applies stored router calibration or reuses stale confidence as a
// probability. All-padding inputs fail rather than producing softmax NaNs.
func (m *Model) Head(ids []int, kind HeadKind, opts Options) (output []float32, err error) {
	defer recoverWork(&err)
	if err = m.validateTokens(ids); err != nil {
		return nil, err
	}
	if opts, err = m.resolveOptions(opts); err != nil {
		return nil, err
	}
	if err = opts.Quant.validate(m.config.Generation); err != nil {
		return nil, err
	}
	if m.config.Generation == 2 && opts.Quant != nil {
		return nil, fmt.Errorf("needle: Needle2 head quantization is not qualified; use FP32")
	}
	_, _, _, err = m.headGeometry(kind)
	if err != nil {
		return nil, err
	}
	e := m.execution(false, opts)
	e.t.reserve(int64(len(ids)) + int64(m.config.Layers+1)*8)
	e.keep = make([]bool, len(ids))
	count := 0
	for i, id := range ids {
		e.keep[i] = id != m.config.PadID
		if e.keep[i] {
			count++
		}
	}
	if count == 0 {
		return nil, fmt.Errorf("needle: auxiliary heads require a non-padding token")
	}
	e.trunk(ids, true)
	result := e.probeHead(kind)
	for _, v := range result.x {
		if !finite(v) {
			return nil, fmt.Errorf("needle: nonfinite head output")
		}
	}
	return result.x, nil
}
func (e *execution) probeHead(kind HeadKind) *value {
	if e.m.config.Generation == 2 {
		return e.probeHeadV2(kind)
	}
	m, t := e.m, e.t
	k, q, _, _ := m.headGeometry(kind)
	l, d := m.config.Layers+1, m.config.DModel
	prefix := string(kind) + "_head/"
	probes := e.headWeight(prefix + "probes")
	gain := e.param(prefix+"gain", -1)
	query := e.headWeight(prefix + "query")
	bias := e.param(prefix+"row_bias", -1)
	pooled := t.alloc(l*k, d)
	scale := float32(1 / math.Sqrt(float64(d)))
	for layer, cell := range e.cells {
		p := t.slice(probes, layer*k*d, k, d)
		scores := t.scale(t.mm(p, cell, true), scale)
		scores = e.maskProbeScores(scores)
		r := t.rms(t.mm(t.softmax(scores, false, 0), cell, false))
		for probe := 0; probe < k; probe++ {
			g := gain.x[layer*k+probe]
			dst := pooled.x[(layer*k+probe)*d : (layer*k+probe+1)*d]
			simd.Saxpy(g, r.x[probe*d:(probe+1)*d], dst)
		}
		if t.train {
			layer, r := layer, r
			t.record(func() {
				for probe := 0; probe < k; probe++ {
					grad := pooled.g[(layer*k+probe)*d : (layer*k+probe+1)*d]
					simd.Saxpy(gain.x[layer*k+probe], grad, r.g[probe*d:(probe+1)*d])
					gain.g[layer*k+probe] += simd.Sdot(grad, r.x[probe*d:(probe+1)*d])
				}
			})
		}
	}
	scores := t.scale(t.mm(query, pooled, true), scale)
	scores = t.add(scores, t.slice(bias, 0, scores.r, scores.c))
	rows := t.mm(t.softmax(scores, false, 0), pooled, false)
	flat := t.slice(rows, 0, 1, q*d)
	proj := e.headWeight(prefix + "proj/kernel")
	result := t.mm(flat, proj, false)
	if kind == Embedding {
		return t.unitHead(result)
	} else {
		result = t.add(result, e.param(prefix+"proj/bias", -1))
	}
	return result
}

func (t *tape) unitHead(result *value) *value {
	var sum float32
	for _, v := range result.x {
		sum += v * v
	}
	norm := float32(math.Sqrt(float64(sum + 1e-12)))
	raw := result
	result = t.alloc(raw.r, raw.c)
	for i := range result.x {
		result.x[i] = raw.x[i] / norm
	}
	if t.train {
		normalized := result
		t.record(func() {
			dot := simd.Sdot(normalized.g, normalized.x)
			for i, g := range normalized.g {
				raw.g[i] += (g - normalized.x[i]*dot) / norm
			}
		})
	}
	return result
}
func (e *execution) maskProbeScores(scores *value) *value {
	t := e.t
	o := t.alloc(scores.r, scores.c)
	copy(o.x, scores.x)
	for row := 0; row < scores.r; row++ {
		for i, keep := range e.keep {
			if !keep {
				o.x[row*scores.c+i] = float32(math.Inf(-1))
			}
		}
	}
	if t.train {
		t.record(func() {
			for row := 0; row < scores.r; row++ {
				for i, keep := range e.keep {
					if keep {
						scores.g[row*scores.c+i] += o.g[row*scores.c+i]
					}
				}
			}
		})
	}
	return o
}
