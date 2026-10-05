package pockettts

import (
	"fmt"
	"math"
	"sort"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// TransformerTrainingTape is a stateless exact-length transformer forward/VJP
// graph with named parameter gradients. All transformer arithmetic is in tape
// operations; host work constructs RoPE tables, admits storage and decodes owned
// outputs. It is transfer-inclusive, with no persistent model/optimiser state.
// Caller must keep the model immutable throughout graph construction/execution.
type TransformerTrainingTape struct {
	Tape             train.Tape
	ParameterNames   []string
	ResidentBindings map[string]train.ResidentTapeBinding
	rows, width      int
}
type TransformerTrainingResult struct {
	Output, DSequence []float32
	Parameters        map[string][]float32
}
type transformerTapeGraph struct {
	tape             train.Tape
	constants        map[int][]float32
	zeros            map[int]bool
	params           map[string]int
	borrowParameters bool
	zeroPool         trainingZeroPool
	borrowed         map[int]bool
	views            map[transformerViewKey]int
	viewSources      map[int]int
	forwardOnly      bool
	zeroSources      map[transformerViewKey]int
	transientZeros   bool // private shared graph: zero fill participates in lifetime reuse
	layerOffset      int  // bounded private stage preserves original parameter names
}
type transformerViewKey struct{ source, rank, a, b, c int }
type transformerGraphNorm struct{ x, w, b, stats int }
type transformerGraphLayer struct {
	input, normalized1, qkv, qr, kr, v, prob, attention, projected, residual, normalized2, pre, gelu, down int
	norm1, norm2                                                                                           transformerGraphNorm
	inWeight, outWeight, fc1Weight, fc2Weight, scale1, scale2                                              int
}

func (g *transformerTapeGraph) tensor(shape []int) int {
	id := len(g.tape.Tensors)
	g.tape.Tensors = append(g.tape.Tensors, train.TensorSpec{Shape: append([]int(nil), shape...)})
	return id
}
func (g *transformerTapeGraph) constant(shape []int, x []float32) int {
	id := g.tensor(shape)
	g.constants[id] = x
	return id
}
func (g *transformerTapeGraph) zero(shape ...int) int {
	if len(shape) >= 1 && len(shape) <= 3 {
		key := transformerViewKey{rank: len(shape), a: shape[0]}
		if len(shape) > 1 {
			key.b = shape[1]
		}
		if len(shape) > 2 {
			key.c = shape[2]
		}
		if id, ok := g.zeroSources[key]; ok {
			return id
		}
		var id int
		if g.transientZeros {
			spec := train.TensorSpec{Shape: shape}
			n, err := spec.Elements()
			if err == nil && n > 1 {
				one := g.zero(1)
				fill := g.command("vector", []int{one, one}, []int{n}, 2)
				// Keep each computed zero local to its consumers; interning it
				// across forward/reverse would extend its live slot to tape end.
				return g.reshape(fill, shape...)
			} else {
				id = g.uniqueZero(shape...)
			}
		} else {
			id = g.uniqueZero(shape...)
		}
		if g.zeroSources == nil {
			g.zeroSources = map[transformerViewKey]int{}
		}
		g.zeroSources[key] = id
		return id
	}
	return g.uniqueZero(shape...)
}

// Parameter-gradient placeholders retain distinct declared root identities.
func (g *transformerTapeGraph) uniqueZero(shape ...int) int {
	id := g.tensor(shape)
	g.zeros[id] = true
	return id
}

// Borrow only an existing admitted positive-zero extent, never grow the pool.
func (g *transformerTapeGraph) workspacePlaceholder(n int, pool *trainingZeroPool) (int, error) {
	id := g.uniqueZero(n)
	if pool != nil {
		if values, ok := (*pool)[n]; ok {
			if len(values) != n || !zeroTrainingSeed(values) {
				return 0, fmt.Errorf("workspace placeholder extent/value changed")
			}
			g.constants[id] = values
			delete(g.zeros, id)
			if g.borrowed == nil {
				g.borrowed = map[int]bool{}
			}
			g.borrowed[id] = true
		}
	}
	return id, nil
}

func (g *transformerTapeGraph) param(name string, shape []int, x []float32) int {
	if id, ok := g.params[name]; ok {
		return id
	}
	id := g.constant(shape, x)
	g.params[name] = id
	if g.borrowParameters {
		if g.borrowed == nil {
			g.borrowed = map[int]bool{}
		}
		g.borrowed[id] = true
	}
	return id
}

// Resident construction borrows only named parameter constants read-only for
// synchronous binding/admission. Other constants retain public owned copies.
func (g *transformerTapeGraph) constantValues(id int, x []float32) []float32 {
	if g.borrowed[id] {
		return x
	}
	return append([]float32(nil), x...)
}

// Zero constants remain distinct tensor IDs; only private read-only host
// source storage is shared. Native allocations and source ceilings are unchanged.
func (g *transformerTapeGraph) zeroValues(n int) []float32 {
	if g.borrowParameters {
		return g.zeroPool.values(n)
	}
	return make([]float32, n)
}
func (g *transformerTapeGraph) command(op string, inputs []int, shape []int, mode uint32) int {
	id := g.tensor(shape)
	g.tape.Commands = append(g.tape.Commands, train.Command{Op: op, Inputs: inputs, Output: id, Mode: mode})
	return id
}
func (g *transformerTapeGraph) reshape(x int, shape ...int) int {
	if x >= 0 && x < len(g.tape.Tensors) && len(shape) >= 1 && len(shape) <= 3 {
		old := g.tape.Tensors[x].Shape
		same := len(old) == len(shape)
		if same {
			for i, d := range shape {
				if old[i] != d {
					same = false
					break
				}
			}
		}
		if same {
			return x
		}
		if g.viewSources == nil {
			g.viewSources = map[int]int{}
			for _, c := range g.tape.Commands {
				if c.Op == "reshape" && len(c.Inputs) == 1 {
					root := c.Inputs[0]
					if previous, ok := g.viewSources[root]; ok {
						root = previous
					}
					g.viewSources[c.Output] = root
				}
			}
		}
		root := x
		if previous, ok := g.viewSources[x]; ok {
			root = previous
		}
		base := g.tape.Tensors[root].Shape
		same = len(base) == len(shape)
		if same {
			for i, d := range shape {
				if base[i] != d {
					same = false
					break
				}
			}
		}
		if same {
			return root
		}
		key := transformerViewKey{source: root, rank: len(shape), a: shape[0]}
		if len(shape) > 1 {
			key.b = shape[1]
		}
		if len(shape) > 2 {
			key.c = shape[2]
		}
		if id, ok := g.views[key]; ok {
			return id
		}
		id := g.command("reshape", []int{root}, shape, 0)
		g.viewSources[id] = root
		if g.views == nil {
			g.views = map[transformerViewKey]int{}
		}
		g.views[key] = id
		return id
	}
	return g.command("reshape", []int{x}, shape, 0)
}
func (g *transformerTapeGraph) flat(x int) int {
	n, _ := g.tape.Tensors[x].Elements()
	return g.reshape(x, n)
}
func (g *transformerTapeGraph) sum(a, b int, shape ...int) int {
	af, bf := g.flat(a), g.flat(b)
	n, _ := g.tape.Tensors[af].Elements()
	id := g.command("vector", []int{af, bf}, []int{n}, 0)
	g.tape.Commands[len(g.tape.Commands)-1].Scale = 1
	return g.reshape(id, shape...)
}
func (g *transformerTapeGraph) row(a, b, rows, width, parts, slot int, mode uint32) int {
	_, _, n, _ := train.RowShape(rows, width, parts, slot, mode)
	id := g.command("row", []int{g.flat(a), g.flat(b)}, []int{n}, mode)
	c := &g.tape.Commands[len(g.tape.Commands)-1]
	c.Rows, c.Width, c.Parts, c.Slot = rows, width, parts, slot
	return id
}
func (g *transformerTapeGraph) linear(prefix string, l LinearF32, x, rows int) (int, int) {
	w := g.param(prefix+".weight", []int{l.Out, l.In}, l.Weight)
	id := g.command("linear-forward", []int{x, g.zero(rows, l.In), w, g.zero(l.Out)}, []int{rows, l.Out}, 0)
	return id, w
}
func (g *transformerTapeGraph) linearBack(x, w, seed, rows int) (int, int) {
	s := g.tape.Tensors[w].Shape
	k, n := s[1], s[0]
	if g.transientZeros {
		// Primal input VJP reads only weight and seed. Delay tangent-zero
		// fills until the parameter VJP that actually consumes them.
		dx := g.command("linear-backward", []int{x, x, w, seed, seed}, []int{rows, k}, 0)
		inputs := []int{x, g.zero(rows, k), w, seed, g.zero(rows, n)}
		return dx, g.command("linear-backward", inputs, []int{n, k}, 2)
	}
	inputs := []int{x, g.zero(rows, k), w, seed, g.zero(rows, n)}
	return g.command("linear-backward", inputs, []int{rows, k}, 0), g.command("linear-backward", inputs, []int{n, k}, 2)
}
func (g *transformerTapeGraph) norm(prefix string, x, rows, width int, w, b []float32) (int, transformerGraphNorm) {
	wi, bi := g.param(prefix+".weight", []int{width}, w), g.param(prefix+".bias", []int{width}, b)
	xt := g.zero(rows, width)
	stats := g.command("norm-forward", []int{x, xt, wi, bi, g.zero(rows, 5)}, []int{rows, 5}, 0)
	g.tape.Commands[len(g.tape.Commands)-1].Epsilon = 1e-5
	out := g.command("norm-forward", []int{x, xt, wi, bi, stats}, []int{rows, width}, 1)
	g.tape.Commands[len(g.tape.Commands)-1].Epsilon = 1e-5
	return out, transformerGraphNorm{x: x, w: wi, b: bi, stats: stats}
}
func (g *transformerTapeGraph) normBack(norm transformerGraphNorm, seed, rows, width int) (int, int, int) {
	inputs := []int{norm.x, g.zero(rows, width), norm.w, seed, g.zero(rows, width), norm.stats}
	return g.command("norm-backward", inputs, []int{rows, width}, 0), g.command("norm-backward", inputs, []int{width}, 2), g.command("norm-backward", inputs, []int{width}, 3)
}
func (g *transformerTapeGraph) attention(op string, inputs []int, shape []int, mode uint32, contextRows int, scale float32) int {
	id := g.command(op, inputs, shape, mode)
	c := &g.tape.Commands[len(g.tape.Commands)-1]
	c.Offset, c.Scale = contextRows, scale
	return id
}

func (m *TransformerCPU) BuildTransformerTrainingTape(sequence, dOutput []float32, rows int, maxBytes int64) (TransformerTrainingTape, error) {
	return m.buildTransformerTrainingTape(sequence, dOutput, rows, maxBytes, false)
}
func (m *TransformerCPU) buildTransformerTrainingTape(sequence, dOutput []float32, rows int, maxBytes int64, borrowParameters bool) (TransformerTrainingTape, error) {
	var result TransformerTrainingTape
	if maxBytes <= 0 {
		return result, fmt.Errorf("transformer tape byte ceiling")
	}
	if err := validateTrainableTransformer(m, sequence, dOutput, rows); err != nil {
		return result, err
	}
	if err := preflightTransformerTrainingTape(m, rows); err != nil {
		return result, err
	}
	g := &transformerTapeGraph{borrowParameters: borrowParameters, constants: map[int][]float32{}, zeros: map[int]bool{}, params: map[string]int{}}
	input := g.constant([]int{rows, m.Width}, sequence)
	seed := g.constant([]int{rows, m.Width}, dOutput)
	output, seed, gradients, err := m.appendTransformerTrainingGraph(g, input, seed, rows)
	if err != nil {
		return result, err
	}
	g.tape.Outputs = []int{output, seed}
	for name := range g.params {
		result.ParameterNames = append(result.ParameterNames, name)
	}
	sort.Strings(result.ParameterNames)
	result.ResidentBindings = map[string]train.ResidentTapeBinding{}
	for _, name := range result.ParameterNames {
		id, ok := gradients[g.params[name]]
		if !ok {
			return result, fmt.Errorf("missing transformer tape gradient %s", name)
		}
		gradient := g.flat(id)
		result.ResidentBindings[name] = train.ResidentTapeBinding{Parameter: g.params[name], Gradient: gradient}
		g.tape.Outputs = append(g.tape.Outputs, gradient)
	}
	// Admit all logical storage and constant lengths/finiteness before tensor
	// materialisation. Metadata and one bounded RoPE table precede this gate.
	if len(g.tape.Tensors) > 4096 || len(g.tape.Commands) > 4096 {
		return result, fmt.Errorf("transformer tape graph ceiling")
	}
	var bytes int64
	for id, s := range g.tape.Tensors {
		n, err := s.Elements()
		if err != nil {
			return result, err
		}
		bytes += int64(n) * 4
		if bytes > maxBytes {
			return result, fmt.Errorf("transformer tape byte ceiling")
		}
		if x, ok := g.constants[id]; ok && (len(x) != n || !finiteF32(x)) {
			return result, fmt.Errorf("transformer tape constant%d shape/nonfinite", id)
		}
	}
	for id, s := range g.tape.Tensors {
		if x, ok := g.constants[id]; ok {
			g.tape.Tensors[id].Values = g.constantValues(id, x)
		} else if g.zeros[id] {
			n, _ := s.Elements()
			g.tape.Tensors[id].Values = g.zeroValues(n)
		}
	}
	if err := g.tape.Validate(maxBytes); err != nil {
		return result, err
	}
	result.Tape, result.rows, result.width = g.tape, rows, m.Width
	return result, nil
}
func (t TransformerTrainingTape) Decode(values [][]float32) (TransformerTrainingResult, error) {
	var result TransformerTrainingResult
	if t.rows <= 0 || t.width <= 0 || len(values) != len(t.Tape.Outputs) || len(values) != 2+len(t.ParameterNames) {
		return result, fmt.Errorf("transformer tape output count")
	}
	for i, x := range values {
		id := t.Tape.Outputs[i]
		if id < 0 || id >= len(t.Tape.Tensors) {
			return result, fmt.Errorf("transformer tape invalid output id")
		}
		n, err := t.Tape.Tensors[id].Elements()
		if err != nil || len(x) != n || !finiteF32(x) {
			return result, fmt.Errorf("transformer tape output%d shape/nonfinite", i)
		}
	}
	result.Output, result.DSequence = append([]float32(nil), values[0]...), append([]float32(nil), values[1]...)
	result.Parameters = map[string][]float32{}
	for i, name := range t.ParameterNames {
		result.Parameters[name] = append([]float32(nil), values[i+2]...)
	}
	return result, nil
}

// Preflight is shared by standalone and graph-composed training.
func preflightTransformerTrainingTape(m *TransformerCPU, rows int) error {
	// Preflight all kernel dimensions before graph metadata or tensor copies.
	if err := train.RoPEShape(rows, m.Heads, m.HeadDim, 0); err != nil {
		return err
	}
	if len(m.Layers) > 16 {
		return fmt.Errorf("transformer tape layer ceiling")
	}
	if err := train.DualNormShape(rows, m.Width, 0, 0); err != nil {
		return err
	}
	for _, l := range m.Layers {
		for _, linear := range []LinearF32{l.InProjection, l.OutProjection, l.FC1, l.FC2} {
			if err := train.LinearShape(rows, linear.In, linear.Out); err != nil {
				return err
			}
		}
	}
	return nil
}

// Append a forward/VJP graph to an admitted input and seed. Constants are
// materialised only after admission of the complete source graph.
func (m *TransformerCPU) appendTransformerTrainingGraph(g *transformerTapeGraph, input, seed, rows int) (int, int, map[int]int, error) {
	return m.appendTransformerTrainingGraphSeed(g, input, seed, rows, nil)
}

// A composed loss may replace the backward seed after forward, before any
// transformer VJP command. Standalone callers keep their exact seed arithmetic.
func (m *TransformerCPU) appendTransformerTrainingGraphSeed(g *transformerTapeGraph, input, seed, rows int, seedGraph func(int, int) int) (int, int, map[int]int, error) {
	contextRows := min(m.Context, rows)
	width, heads, dim := m.Width, m.Heads, m.HeadDim
	shape, dense := []int{rows, width}, []int{heads, rows, rows}
	headShape := []int{rows, heads, dim}
	hidden := input
	table := make([]float32, rows*dim)
	if err := train.RoPETableInto(table, rows, dim, 0, m.MaxPeriod); err != nil {
		return 0, 0, nil, err
	}
	tableID := g.constant([]int{rows, dim / 2, 2}, table)
	layers := make([]transformerGraphLayer, len(m.Layers))
	scale := float32(1 / math.Sqrt(float64(dim)))
	for i, l := range m.Layers {
		p := fmt.Sprintf("flow_lm.transformer.layers.%d", i+g.layerOffset)
		t := &layers[i]
		t.input = hidden
		t.scale1, t.scale2 = -1, -1
		t.normalized1, t.norm1 = g.norm(p+".norm1", hidden, rows, width, l.Norm1Weight, l.Norm1Bias)
		t.qkv, t.inWeight = g.linear(p+".self_attn.in_proj", l.InProjection, t.normalized1, rows)
		unused := g.zero(width)
		q := g.row(t.qkv, unused, rows, width, 3, 0, 0)
		k := g.row(t.qkv, unused, rows, width, 3, 1, 0)
		v := g.row(t.qkv, unused, rows, width, 3, 2, 0)
		t.qr = g.command("rope", []int{g.reshape(q, headShape...), tableID}, headShape, 0)
		t.kr = g.command("rope", []int{g.reshape(k, headShape...), tableID}, headShape, 0)
		t.v = g.reshape(v, headShape...)
		scratch := g.zero(dense...)
		scores := g.attention("attention-forward", []int{t.qr, t.kr, t.v, scratch}, dense, 0, contextRows, scale)
		t.prob = g.attention("attention-forward", []int{t.qr, t.kr, t.v, scores}, dense, 1, contextRows, scale)
		attention := g.attention("attention-forward", []int{t.qr, t.kr, t.v, t.prob}, headShape, 2, contextRows, scale)
		t.attention = g.reshape(attention, shape...)
		t.projected, t.outWeight = g.linear(p+".self_attn.out_proj", l.OutProjection, t.attention, rows)
		update := t.projected
		if l.LayerScale1 != nil {
			t.scale1 = g.param(p+".layer_scale_1.scale", []int{width}, l.LayerScale1)
			update = g.reshape(g.row(t.projected, t.scale1, rows, width, 1, 0, 2), shape...)
		}
		t.residual = g.sum(hidden, update, shape...)
		t.normalized2, t.norm2 = g.norm(p+".norm2", t.residual, rows, width, l.Norm2Weight, l.Norm2Bias)
		t.pre, t.fc1Weight = g.linear(p+".linear1", l.FC1, t.normalized2, rows)
		t.gelu = g.reshape(g.command("gelu", []int{g.flat(t.pre), g.zero(rows * l.FC1.Out)}, []int{rows * l.FC1.Out}, 0), rows, l.FC1.Out)
		t.down, t.fc2Weight = g.linear(p+".linear2", l.FC2, t.gelu, rows)
		update = t.down
		if l.LayerScale2 != nil {
			t.scale2 = g.param(p+".layer_scale_2.scale", []int{width}, l.LayerScale2)
			update = g.reshape(g.row(t.down, t.scale2, rows, width, 1, 0, 2), shape...)
		}
		hidden = g.sum(t.residual, update, shape...)
	}
	var final transformerGraphNorm
	if m.FinalWeight != nil {
		hidden, final = g.norm("flow_lm.out_norm", hidden, rows, width, m.FinalWeight, m.FinalBias)
	}
	output := hidden
	if g.forwardOnly {
		return output, 0, nil, nil
	}
	if seedGraph != nil {
		seed = seedGraph(output, seed)
	}
	gradients := map[int]int{}
	if m.FinalWeight != nil {
		var dw, db int
		seed, dw, db = g.normBack(final, seed, rows, width)
		gradients[final.w], gradients[final.b] = dw, db
	}
	for i := len(layers) - 1; i >= 0; i-- {
		t, l := layers[i], m.Layers[i]
		downSeed := seed
		if t.scale2 >= 0 {
			gradients[t.scale2] = g.row(t.down, seed, rows, width, 1, 0, 3)
			downSeed = g.reshape(g.row(seed, t.scale2, rows, width, 1, 0, 2), shape...)
		}
		dgelu, dw2 := g.linearBack(t.gelu, t.fc2Weight, downSeed, rows)
		gradients[t.fc2Weight] = dw2
		dpre := g.reshape(g.command("gelu", []int{g.flat(t.pre), g.flat(dgelu)}, []int{rows * l.FC1.Out}, 1), rows, l.FC1.Out)
		dnorm2, dw1 := g.linearBack(t.normalized2, t.fc1Weight, dpre, rows)
		gradients[t.fc1Weight] = dw1
		dn2, dn2w, dn2b := g.normBack(t.norm2, dnorm2, rows, width)
		gradients[t.norm2.w], gradients[t.norm2.b] = dn2w, dn2b
		residualSeed := g.sum(seed, dn2, shape...)
		projectionSeed := residualSeed
		if t.scale1 >= 0 {
			gradients[t.scale1] = g.row(t.projected, residualSeed, rows, width, 1, 0, 3)
			projectionSeed = g.reshape(g.row(residualSeed, t.scale1, rows, width, 1, 0, 2), shape...)
		}
		da, dwout := g.linearBack(t.attention, t.outWeight, projectionSeed, rows)
		gradients[t.outWeight] = dwout
		da = g.reshape(da, headShape...)
		back := func(aux int, outShape []int, mode uint32) int {
			return g.attention("attention-backward", []int{t.qr, t.kr, t.v, t.prob, da, aux}, outShape, mode, contextRows, scale)
		}
		aux := t.prob
		if !g.transientZeros {
			aux = g.zero(dense...)
		}
		// Experimental lifetime mode rebinds unused aux to a read-only source.
		// Ordinary graphs keep the previous literal-zero construction.
		dp := back(aux, dense, 0)
		ds := back(dp, dense, 1)
		var dq, dk, dv int
		dv = -1
		if g.transientZeros {
			dq = back(ds, headShape, 2)
			dq = g.command("rope", []int{dq, tableID}, headShape, 1)
		} else {
			dq, dk = back(ds, headShape, 2), back(ds, headShape, 3)
			dv = back(aux, headShape, 4)
			dq = g.command("rope", []int{dq, tableID}, headShape, 1)
			dk = g.command("rope", []int{dk, tableID}, headShape, 1)
		}
		unused := g.zero(width)
		var packed int
		if g.transientZeros {
			qpacked := g.row(dq, unused, rows, width, 3, 0, 1)
			dk = back(ds, headShape, 3)
			dk = g.command("rope", []int{dk, tableID}, headShape, 1)
			kpacked := g.row(dk, unused, rows, width, 3, 1, 1)
			qkpacked := g.sum(qpacked, kpacked, rows, 3*width)
			dv = back(t.prob, headShape, 4)
			vpacked := g.row(dv, unused, rows, width, 3, 2, 1)
			packed = g.sum(qkpacked, vpacked, rows, 3*width)
		} else {
			qpacked := g.row(dq, unused, rows, width, 3, 0, 1)
			kpacked := g.row(dk, unused, rows, width, 3, 1, 1)
			vpacked := g.row(dv, unused, rows, width, 3, 2, 1)
			packed = g.sum(g.sum(qpacked, kpacked, rows, 3*width), vpacked, rows, 3*width)
		}
		dnorm1, dwin := g.linearBack(t.normalized1, t.inWeight, packed, rows)
		gradients[t.inWeight] = dwin
		dn1, dn1w, dn1b := g.normBack(t.norm1, dnorm1, rows, width)
		gradients[t.norm1.w], gradients[t.norm1.b] = dn1w, dn1b
		seed = g.sum(residualSeed, dn1, shape...)
	}
	return output, seed, gradients, nil
}
