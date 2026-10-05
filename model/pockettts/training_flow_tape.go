package pockettts

import (
	"fmt"
	"math"
	"sort"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// FlowDualTrainingTape is a single-row, transfer-inclusive GPU flow graph.
// Timestep trigonometric embeddings are immutable caller-computed F32 inputs;
// unless deviceTime is selected by the separate explicit builder. In both paths,
// linear/activation/norm/AdaLN forward and reverse-over-forward run in the tape.
// It is not a complete resident trainer. Outputs include ordinary prediction,
// selected-time JVP, input adjoints and named parameter gradients. Shapes and
// resource ceilings are admitted before any native allocation.
type FlowDualTrainingTape struct {
	Tape                     train.Tape
	ParameterNames           []string
	ResidentBindings         map[string]train.ResidentTapeBinding
	conditionSize, inputSize int
	timeEmbeddings           []flowTapeTime
	outputSize               int
	deviceTime               bool
	inputValue               int // explicit composition edge; standalone arithmetic unchanged
}
type flowTapeTime struct {
	frequency                  []float32
	time, dtime                float32
	valueOutput, tangentOutput int
}
type FlowDualTrainingResult struct {
	Output, Tangent, DCondition, DInput []float32
	DTimes                              []float32
	Parameters                          map[string][]float32
}
type graphDual struct{ v, t int }
type graphNode struct {
	op string
	// All flow AD nodes are unary or binary. Inline inputs avoid a separate
	// operand allocation; unary operations read only the first slot.
	input   [2]graphDual
	output  graphDual
	w, b    int
	kind    uint32
	epsilon float32
	offset  int
	scale   float32
}
type flowTapeGraph struct {
	tape             train.Tape
	nodes            []graphNode
	params           map[string]int
	borrowParameters bool
	ordinaryReverse  bool
	zeroPool         trainingZeroPool
	shapeArena       flowShapeArena
	operandArena     flowOperandArena
	adj              map[int]int
	err              error
}

func (g *flowTapeGraph) tensor(shape []int, values []float32) int {
	id := len(g.tape.Tensors)
	g.tape.Tensors = append(g.tape.Tensors, train.TensorSpec{Shape: g.shapeArena.copy(shape), Values: values})
	return id
}
func (g *flowTapeGraph) constant(values []float32) int {
	return g.tensor([]int{len(values)}, append([]float32(nil), values...))
}
func (g *flowTapeGraph) zero(n int) int {
	if g.borrowParameters {
		return g.tensor([]int{n}, g.zeroPool.values(n))
	}
	return g.constant(make([]float32, n))
}
func (g *flowTapeGraph) shape(id int) []int { return g.tape.Tensors[id].Shape }
func (g *flowTapeGraph) elements(id int) int {
	n, err := g.tape.Tensors[id].Elements()
	if err != nil {
		g.err = err
	}
	return n
}
func (g *flowTapeGraph) command(op string, inputs []int, shape []int, kind, mode uint32, offset int, scale, epsilon float32) int {
	id := g.tensor(shape, nil)
	g.tape.Commands = append(g.tape.Commands, train.Command{Op: op, Inputs: g.operandArena.copy(inputs), Output: id, Kind: kind, Mode: mode, Offset: offset, Scale: scale, Epsilon: epsilon})
	return id
}
func (g *flowTapeGraph) reshape(id int, shape ...int) int {
	return g.command("reshape", []int{id}, shape, 0, 0, 0, 0, 0)
}
func (g *flowTapeGraph) dual(values []float32) graphDual {
	return graphDual{g.constant(values), g.zero(len(values))}
}
func (g *flowTapeGraph) parameter(name string, shape []int, values []float32) int {
	if id, ok := g.params[name]; ok {
		return id
	}
	if !g.borrowParameters {
		values = append([]float32(nil), values...)
	}
	id := g.tensor(shape, values)
	g.params[name] = id
	return id
}
func (g *flowTapeGraph) linear(prefix string, l LinearF32, x graphDual) graphDual {
	w := g.parameter(prefix+".weight", []int{l.Out, l.In}, l.Weight)
	bias := g.zero(l.Out)
	if l.Bias != nil {
		bias = g.parameter(prefix+".bias", []int{l.Out}, l.Bias)
	}
	xv, xt := g.reshape(x.v, 1, l.In), g.reshape(x.t, 1, l.In)
	v := g.command("linear-forward", []int{xv, xt, w, bias}, []int{1, l.Out}, 0, 0, 0, 0, 0)
	t := g.command("linear-forward", []int{xv, xt, w, bias}, []int{1, l.Out}, 0, 1, 0, 0, 0)
	out := graphDual{g.reshape(v, l.Out), g.reshape(t, l.Out)}
	g.nodes = append(g.nodes, graphNode{op: "linear", input: [2]graphDual{x}, output: out, w: w, b: bias})
	return out
}
func (g *flowTapeGraph) silu(x graphDual) graphDual {
	n := g.elements(x.v)
	out := graphDual{g.command("silu-forward", []int{x.v, x.t}, []int{n}, 0, 0, 0, 0, 0), g.command("silu-forward", []int{x.v, x.t}, []int{n}, 0, 1, 0, 0, 0)}
	g.nodes = append(g.nodes, graphNode{op: "silu", input: [2]graphDual{x}, output: out})
	return out
}
func (g *flowTapeGraph) norm(prefix string, x graphDual, weight, bias []float32, epsilon float32, kind uint32) graphDual {
	n := g.elements(x.v)
	w, b := g.constant(onesTraining(n)), g.zero(n)
	if weight != nil {
		w = g.parameter(prefix+".weight", []int{n}, weight)
	}
	if bias != nil {
		b = g.parameter(prefix+".bias", []int{n}, bias)
	}
	xv, xt := g.reshape(x.v, 1, n), g.reshape(x.t, 1, n)
	scratch := g.tensor([]int{1, 5}, make([]float32, 5))
	stats := g.command("norm-forward", []int{xv, xt, w, b, scratch}, []int{1, 5}, kind, 0, 0, 0, epsilon)
	v := g.command("norm-forward", []int{xv, xt, w, b, stats}, []int{1, n}, kind, 1, 0, 0, epsilon)
	t := g.command("norm-forward", []int{xv, xt, w, b, stats}, []int{1, n}, kind, 2, 0, 0, epsilon)
	out := graphDual{g.reshape(v, n), g.reshape(t, n)}
	g.nodes = append(g.nodes, graphNode{op: "norm", input: [2]graphDual{x}, output: out, w: w, b: b, kind: kind, epsilon: epsilon, offset: stats})
	return out
}
func onesTraining(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}
func (g *flowTapeGraph) mod(a, b graphDual, kind uint32) graphDual {
	n := g.elements(a.v)
	out := graphDual{g.command("mod-forward", []int{a.v, a.t, b.v, b.t}, []int{n}, kind, 0, 0, 0, 0), g.command("mod-forward", []int{a.v, a.t, b.v, b.t}, []int{n}, kind, 1, 0, 0, 0)}
	g.nodes = append(g.nodes, graphNode{op: "mod", input: [2]graphDual{a, b}, output: out, kind: kind})
	return out
}
func (g *flowTapeGraph) add(a, b graphDual, scale float32) graphDual {
	n := g.elements(a.v)
	out := graphDual{g.command("vector", []int{a.v, b.v}, []int{n}, 0, 0, 0, scale, 0), g.command("vector", []int{a.t, b.t}, []int{n}, 0, 0, 0, scale, 0)}
	g.nodes = append(g.nodes, graphNode{op: "add", input: [2]graphDual{a, b}, output: out, scale: scale})
	return out
}
func (g *flowTapeGraph) slice(x graphDual, offset, n int) graphDual {
	scratch := g.zero(1)
	out := graphDual{g.command("vector", []int{x.v, scratch}, []int{n}, 0, 1, offset, 0, 0), g.command("vector", []int{x.t, scratch}, []int{n}, 0, 1, offset, 0, 0)}
	g.nodes = append(g.nodes, graphNode{op: "slice", input: [2]graphDual{x}, output: out, offset: offset})
	return out
}
func (g *flowTapeGraph) gradient(id int) int {
	if grad, ok := g.adj[id]; ok {
		return grad
	}
	grad := g.zero(g.elements(id))
	g.adj[id] = grad
	return grad
}
func (g *flowTapeGraph) accumulate(id, contribution int) {
	if previous, ok := g.adj[id]; ok {
		g.adj[id] = g.command("vector", []int{previous, contribution}, []int{g.elements(id)}, 0, 0, 0, 1, 0)
	} else {
		g.adj[id] = contribution
	}
}
func (g *flowTapeGraph) reverse(out graphDual, gv, gt []float32) {
	g.reverseIDs(out, g.constant(gv), g.constant(gt))
}
func (g *flowTapeGraph) reverseIDs(out graphDual, gv, gt int) {
	g.adj[out.v], g.adj[out.t] = gv, gt
	g.reserveReverseMetadata()
	for i := len(g.nodes) - 1; i >= 0; i-- {
		node := g.nodes[i]
		v, t := g.gradient(node.output.v), g.gradient(node.output.t)
		n := g.elements(node.output.v)
		switch node.op {
		case "linear":
			x := node.input[0]
			k := g.elements(x.v)
			wshape := g.shape(node.w)
			xv, xt := g.reshape(x.v, 1, k), g.reshape(x.t, 1, k)
			sv, st := g.reshape(v, 1, n), g.reshape(t, 1, n)
			inputs := []int{xv, xt, node.w, sv, st}
			for mode, id := range []int{x.v, x.t} {
				grad := g.command("linear-backward", inputs, []int{1, k}, 0, uint32(mode), 0, 0, 0)
				g.accumulate(id, g.reshape(grad, k))
			}
			dw := g.command("linear-backward", inputs, wshape, 0, 2, 0, 0, 0)
			g.accumulate(node.w, g.reshape(dw, k*n))
			db := g.command("linear-backward", inputs, []int{n}, 0, 3, 0, 0, 0)
			g.accumulate(node.b, db)
		case "silu":
			x := node.input[0]
			if g.ordinaryReverse {
				// Ordinary CPU reverse rounds (seed*sigmoid)*factor.
				// Reuse the tangent kernel for this association; mixed reverse
				// retains seed*(sigmoid*factor) and its second-order term.
				id := g.command("silu-forward", []int{x.v, v}, []int{n}, 0, 1, 0, 0, 0)
				g.accumulate(x.v, id)
				continue
			}
			for mode, id := range []int{x.v, x.t} {
				grad := g.command("silu-backward", []int{x.v, x.t, v, t}, []int{n}, 0, uint32(mode), 0, 0, 0)
				g.accumulate(id, grad)
			}
		case "norm":
			x := node.input[0]
			xv, xt := g.reshape(x.v, 1, n), g.reshape(x.t, 1, n)
			sv, st := g.reshape(v, 1, n), g.reshape(t, 1, n)
			inputs := []int{xv, xt, node.w, sv, st, node.offset}
			for mode, id := range []int{x.v, x.t} {
				grad := g.command("norm-backward", inputs, []int{1, n}, node.kind, uint32(mode), 0, 0, 0)
				g.accumulate(id, g.reshape(grad, n))
			}
			dw := g.command("norm-backward", inputs, []int{n}, node.kind, 2, 0, 0, 0)
			g.accumulate(node.w, dw)
			if node.kind == 0 {
				db := g.command("norm-backward", inputs, []int{n}, 0, 3, 0, 0, 0)
				g.accumulate(node.b, db)
			}
		case "mod":
			a, b := node.input[0], node.input[1]
			inputs := []int{a.v, a.t, b.v, b.t, v, t}
			for mode, id := range []int{a.v, a.t, b.v, b.t} {
				grad := g.command("mod-backward", inputs, []int{g.elements(id)}, node.kind, uint32(mode), 0, 0, 0)
				g.accumulate(id, grad)
			}
		case "add":
			a, b := node.input[0], node.input[1]
			g.accumulate(a.v, v)
			g.accumulate(a.t, t)
			zero := g.zero(n)
			g.accumulate(b.v, g.command("vector", []int{zero, v}, []int{n}, 0, 0, 0, node.scale, 0))
			g.accumulate(b.t, g.command("vector", []int{zero, t}, []int{n}, 0, 0, 0, node.scale, 0))
		case "slice":
			x := node.input[0]
			for mode, id := range []int{x.v, x.t} {
				seed := v
				if mode == 1 {
					seed = t
				}
				padded := g.command("vector", []int{seed, g.zero(1)}, []int{g.elements(id)}, 0, 2, node.offset, 0, 0)
				g.accumulate(id, padded)
			}
		}
	}
}

// BuildFlowDualTrainingTape builds exact reverse-over-forward operations with
// an explicit selected time tangent (−1 means all tangents zero). It mutates no
// model parameters. Times use upstream sinusoidal embedding order cos then sin.
func (m *FlowHeadCPU) BuildFlowDualTrainingTape(condition, times, input []float32, timeIndex int, gv, gt []float32, maxBytes int64) (FlowDualTrainingTape, error) {
	return m.buildFlowDualTrainingTape(condition, times, input, timeIndex, gv, gt, maxBytes, false, false)
}

// BuildFlowDeviceTimeTrainingTape computes sinusoidal values, selected-time JVP
// and scalar reverse-over-forward time adjoints on device. Legacy builders keep
// their host-F64-rounded trigonometry and boundary ordering unchanged.
func (m *FlowHeadCPU) BuildFlowDeviceTimeTrainingTape(condition, times, input []float32, timeIndex int, gv, gt []float32, maxBytes int64) (FlowDualTrainingTape, error) {
	return m.buildFlowDualTrainingTape(condition, times, input, timeIndex, gv, gt, maxBytes, true, false)
}
func (m *FlowHeadCPU) buildFlowDualTrainingTape(condition, times, input []float32, timeIndex int, gv, gt []float32, maxBytes int64, deviceTime, borrowParameters bool) (FlowDualTrainingTape, error) {
	return m.buildFlowDualTrainingTapeSeed(condition, times, input, timeIndex, gv, gt, maxBytes, deviceTime, borrowParameters, nil)
}
func (m *FlowHeadCPU) buildFlowDualTrainingTapeSeed(condition, times, input []float32, timeIndex int, gv, gt []float32, maxBytes int64, deviceTime, borrowParameters bool, seedGraph func(*flowTapeGraph, graphDual) (int, int)) (FlowDualTrainingTape, error) {
	return m.buildFlowDualTrainingTapeSources(condition, times, input, timeIndex, gv, gt, maxBytes, deviceTime, borrowParameters, seedGraph, nil)
}
func (m *FlowHeadCPU) buildFlowDualTrainingTapeSources(condition, times, input []float32, timeIndex int, gv, gt []float32, maxBytes int64, deviceTime, borrowParameters bool, seedGraph func(*flowTapeGraph, graphDual) (int, int), external *flowDeviceTapeSources) (FlowDualTrainingTape, error) {
	var result FlowDualTrainingTape
	result.deviceTime = deviceTime
	conditionSize, inputSize, outputSize := len(condition), len(input), len(gv)
	if external == nil {
		if err := validateTrainableFlowHead(m, condition, times, input, gv); err != nil {
			return result, err
		}
		if len(gt) != len(gv) || !finiteF32(gt) {
			return result, fmt.Errorf("flow dual tape seeds/time selection")
		}
	} else {
		if !deviceTime || condition != nil || input != nil || gv != nil || gt != nil {
			return result, fmt.Errorf("device flow sources require shape-only inputs and device time")
		}
		conditionSize, inputSize, outputSize = external.sizes.Condition, external.sizes.Input, external.sizes.Output
		if err := validateTrainableFlowHeadSizes(m, conditionSize, len(times), inputSize, outputSize); err != nil {
			return result, err
		}
		if !finiteF32(times) {
			return result, fmt.Errorf("device flow times nonfinite")
		}
	}
	if timeIndex < -1 || timeIndex >= len(times) {
		return result, fmt.Errorf("flow dual tape seeds/time selection")
	}
	if deviceTime {
		if maxBytes <= 0 {
			return result, fmt.Errorf("device time tape byte ceiling required")
		}
		for i, time := range m.Time {
			if _, err := train.TimeEmbeddingShape(len(time.Frequencies), 0); err != nil {
				return result, err
			}
			for _, f := range time.Frequencies {
				angle := times[i] * f
				if !isFinite(f) || !isFinite(angle) || float32(math.Abs(float64(angle))) > 8192 || !isFinite(f*f) {
					return result, fmt.Errorf("device time frequency/angle ceiling")
				}
			}
		}
	}
	g := &flowTapeGraph{borrowParameters: borrowParameters, ordinaryReverse: timeIndex < 0, params: map[string]int{}, adj: map[int]int{}, tape: train.Tape{Commands: make([]train.Command, 0, flowForwardCommandCapacity(len(m.Time), len(m.Blocks), deviceTime))}}
	var x, cond graphDual
	if external == nil {
		x, cond = g.dual(input), g.dual(condition)
	} else {
		x = graphDual{g.external(inputSize), g.zero(inputSize)}
		cond = graphDual{g.external(conditionSize), g.zero(conditionSize)}
		external.ids.Input, external.ids.Condition = x.v, cond.v
	}
	result.inputValue = x.v
	hidden := g.linear("flow_lm.flow_net.input_proj", m.Input, x)
	combined := g.linear("flow_lm.flow_net.cond_embed", m.Condition, cond)
	var embeddings []graphDual
	for i, time := range m.Time {
		tangent := float32(0)
		if i == timeIndex {
			tangent = 1
		}
		half := len(time.Frequencies)
		var emb graphDual
		if deviceTime {
			f := g.constant(time.Frequencies)
			t := g.constant([]float32{times[i], tangent})
			z := g.zero(2 * half)
			inputs := []int{f, t, z, z}
			emb = graphDual{g.command("time-embedding", inputs, []int{2 * half}, 0, 0, 0, 0, 0), g.command("time-embedding", inputs, []int{2 * half}, 0, 1, 0, 0, 0)}
		} else {
			values, derivatives := make([]float32, half*2), make([]float32, half*2)
			for j, frequency := range time.Frequencies {
				angle := times[i] * frequency
				sin, cos := float32(math.Sin(float64(angle))), float32(math.Cos(float64(angle)))
				values[j], values[half+j] = cos, sin
				derivatives[j], derivatives[half+j] = -sin*frequency*tangent, cos*frequency*tangent
			}
			emb = graphDual{g.constant(values), g.constant(derivatives)}
		}
		embeddings = append(embeddings, emb)
		prefix := fmt.Sprintf("flow_lm.flow_net.time_embed.%d.mlp", i)
		h := g.silu(g.linear(prefix+".0", time.FC1, emb))
		h = g.linear(prefix+".2", time.FC2, h)
		// Name alpha exactly as native checkpoint, no .weight suffix.
		normalized := g.norm(prefix+".3", h, time.RMSWeight, nil, time.RMSEpsilon, 1)
		g.params[prefix+".3.alpha"] = g.params[prefix+".3.weight"]
		delete(g.params, prefix+".3.weight")
		combined = g.add(combined, normalized, 1/float32(len(m.Time)))
		result.timeEmbeddings = append(result.timeEmbeddings, flowTapeTime{frequency: append([]float32(nil), time.Frequencies...), time: times[i], dtime: tangent})
	}
	for i, block := range m.Blocks {
		prefix := fmt.Sprintf("flow_lm.flow_net.res_blocks.%d", i)
		d := len(block.NormWeight)
		mod := g.linear(prefix+".adaLN_modulation.1", block.Modulation, g.silu(combined))
		norm := g.norm(prefix+".in_ln", hidden, block.NormWeight, block.NormBias, block.Epsilon, 0)
		shiftScale := g.slice(mod, 0, 2*d)
		gate := g.slice(mod, 2*d, d)
		transformed := g.mod(norm, shiftScale, 0)
		update := g.linear(prefix+".mlp.2", block.FC2, g.silu(g.linear(prefix+".mlp.0", block.FC1, transformed)))
		gated := g.mod(update, gate, 1)
		next := g.add(hidden, gated, 1)
		// CPU residual JVP is (hiddenT + gateT*update) + gate*updateT.
		// Preserve that association instead of adding a pre-summed gated JVP.
		g.tape.Commands = g.tape.Commands[:len(g.tape.Commands)-1]
		n := g.elements(update.v)
		first := g.command("mod-forward", []int{gate.t, g.zero(n), update.v, g.zero(n)}, []int{n}, 1, 0, 0, 0, 0)
		second := g.command("mod-forward", []int{gate.v, g.zero(n), update.t, g.zero(n)}, []int{n}, 1, 0, 0, 0, 0)
		partial := g.command("vector", []int{hidden.t, first}, []int{n}, 0, 0, 0, 1, 0)
		// Keep the originally declared tangent tensor ID and backward edge.
		g.tape.Commands = append(g.tape.Commands, train.Command{Op: "vector", Inputs: []int{partial, second}, Output: next.t, Scale: 1})
		hidden = next
	}
	prefix := "flow_lm.flow_net.final_layer"
	mod := g.linear(prefix+".adaLN_modulation.1", m.Final.Modulation, g.silu(combined))
	norm := g.norm(prefix+".norm", hidden, nil, nil, m.Final.Epsilon, 0)
	output := g.linear(prefix+".linear", m.Final.Linear, g.mod(norm, mod, 0))
	if seedGraph != nil {
		v, t := seedGraph(g, output)
		g.reverseIDs(output, v, t)
	} else if external != nil {
		external.ids.ValueSeed, external.ids.TangentSeed = g.external(outputSize), g.external(outputSize)
		g.reverseIDs(output, external.ids.ValueSeed, external.ids.TangentSeed)
	} else {
		g.reverse(output, gv, gt)
	}
	g.tape.Outputs = []int{output.v, output.t, g.gradient(cond.v), g.gradient(x.v)}
	result.conditionSize, result.inputSize, result.outputSize = conditionSize, inputSize, outputSize
	for i, emb := range embeddings {
		result.timeEmbeddings[i].valueOutput = len(g.tape.Outputs)
		if deviceTime {
			time := result.timeEmbeddings[i]
			inputs := []int{g.constant(time.frequency), g.constant([]float32{time.time, time.dtime}), g.gradient(emb.v), g.gradient(emb.t)}
			g.tape.Outputs = append(g.tape.Outputs, g.command("time-embedding", inputs, []int{1}, 0, 2, 0, 0, 0))
		} else {
			g.tape.Outputs = append(g.tape.Outputs, g.gradient(emb.v))
			result.timeEmbeddings[i].tangentOutput = len(g.tape.Outputs)
			g.tape.Outputs = append(g.tape.Outputs, g.gradient(emb.t))
		}
	}
	for name := range g.params {
		result.ParameterNames = append(result.ParameterNames, name)
	}
	sort.Strings(result.ParameterNames)
	result.ResidentBindings = map[string]train.ResidentTapeBinding{}
	for _, name := range result.ParameterNames {
		gradient := g.gradient(g.params[name])
		result.ResidentBindings[name] = train.ResidentTapeBinding{Parameter: g.params[name], Gradient: gradient}
		g.tape.Outputs = append(g.tape.Outputs, gradient)
	}
	if g.err != nil {
		return FlowDualTrainingTape{}, g.err
	}
	if err := g.tape.Validate(maxBytes); err != nil {
		return FlowDualTrainingTape{}, err
	}
	result.Tape = g.tape
	return result, nil
}
func (t FlowDualTrainingTape) Decode(values [][]float32) (FlowDualTrainingResult, error) {
	var result FlowDualTrainingResult
	if len(values) != len(t.Tape.Outputs) || len(values) < 4 {
		return result, fmt.Errorf("flow tape result count")
	}
	for i, values := range values {
		n, _ := t.Tape.Tensors[t.Tape.Outputs[i]].Elements()
		if len(values) != n || !finiteF32(values) {
			return result, fmt.Errorf("flow tape result%d shape/nonfinite", i)
		}
	}
	boundary := t.boundarySize()
	result, err := t.decodeBoundary(values[:boundary])
	if err != nil {
		return FlowDualTrainingResult{}, err
	}
	for i, name := range t.ParameterNames {
		result.Parameters[name] = append([]float32(nil), values[boundary+i]...)
	}
	return result, nil
}

// decodeBoundary consumes only input/value/time adjoints; parameter gradients
// may remain resident. No host gradient placeholder or download is required.
func (t FlowDualTrainingTape) boundarySize() int {
	if t.deviceTime {
		return 4 + len(t.timeEmbeddings)
	}
	return 4 + 2*len(t.timeEmbeddings)
}
func (t FlowDualTrainingTape) decodeBoundary(values [][]float32) (FlowDualTrainingResult, error) {
	var result FlowDualTrainingResult
	if len(values) != t.boundarySize() {
		return result, fmt.Errorf("flow boundary count")
	}
	for i, x := range values {
		n, _ := t.Tape.Tensors[t.Tape.Outputs[i]].Elements()
		if len(x) != n || !finiteF32(x) {
			return result, fmt.Errorf("flow boundary%d shape/nonfinite", i)
		}
	}
	own := func(i int) []float32 { return append([]float32(nil), values[i]...) }
	result.Output, result.Tangent, result.DCondition, result.DInput = own(0), own(1), own(2), own(3)
	result.DTimes = make([]float32, len(t.timeEmbeddings))
	result.Parameters = map[string][]float32{}
	for i, time := range t.timeEmbeddings {
		if t.deviceTime {
			result.DTimes[i] = values[time.valueOutput][0]
			continue
		}
		a, b := values[time.valueOutput], values[time.tangentOutput]
		half := len(time.frequency)
		for j, f := range time.frequency {
			angle := time.time * f
			sin, cos := float32(math.Sin(float64(angle))), float32(math.Cos(float64(angle)))
			fc, fs := -sin*f, cos*f
			sc, ss := -cos*f*f, -sin*f*f
			result.DTimes[i] += a[j]*fc + a[half+j]*fs
			result.DTimes[i] += b[j]*sc*time.dtime + b[half+j]*ss*time.dtime
		}
	}
	return result, nil
}
