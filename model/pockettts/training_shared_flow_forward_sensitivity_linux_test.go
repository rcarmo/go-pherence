//go:build linux

package pockettts

import (
	"fmt"
	"math"
	"testing"
)

// Explicit CPU diagnostic backend, with paired control. Backend orchestration
// computes local gradients before outer objective scaling, unlike the default
// scratch path, so compare the paired control as well as saved CPU/Vulkan.
// No runtime, production fallback, parameter update or device execution exists.
type sharedFlowForwardSensitivityBackend struct {
	serial                bool
	shaderNorm            bool
	serialCondition       bool
	normAdjoint           bool
	backwardPolicy        *sharedFlowBackwardPolicy
	normObservation       sharedFlowNormAdjointObservation
	condition             sharedFlowConditionObservation
	calls, ordinary, dual int
	output, tangent       sharedAttentionError
}

func sharedSerialFlowLinear(l LinearF32, x dualVector) dualVector {
	v := sharedSerialLinearForward(l, x.value, 1)
	l.Bias = nil
	return dualVector{value: v, tangent: sharedSerialLinearForward(l, x.tangent, 1)}
}

func sharedFlowRoundedNorm(input dualVector, weight, bias []float32, epsilon float32, varianceOnly bool) (dualVector, float32, float32, float32, float32, float32, float32) {
	n := len(input.value)
	var mean, dm, variance, dv float32
	for i, x := range input.value {
		mean += x
		dm += input.tangent[i]
	}
	mean /= float32(n)
	dm /= float32(n)
	for i, x := range input.value {
		c, dc := x-mean, input.tangent[i]-dm
		variance += float32(c * c)
		dv += float32((2 * c) * dc)
	}
	denom := float32(n)
	if varianceOnly {
		denom = float32(n - 1)
	}
	variance /= denom
	dv /= denom
	inv := float32(1) / float32(math.Sqrt(float64(variance+epsilon)))
	dinv := (((float32(-.5) * inv) * inv) * inv) * dv
	out := dualVector{value: make([]float32, n), tangent: make([]float32, n)}
	for i, x := range input.value {
		v, t := x-mean, input.tangent[i]-dm
		if varianceOnly {
			v, t = x, input.tangent[i]
		}
		w := float32(1)
		if weight != nil {
			w = weight[i]
		}
		if varianceOnly {
			out.value[i] = (v * w) * inv
		} else {
			out.value[i] = (v * inv) * w
			if bias != nil {
				out.value[i] += bias[i]
			}
		}
		out.tangent[i] = w * (t*inv + v*dinv)
	}
	return out, mean, dm, variance, dv, inv, dinv
}

func sharedSerialFlowDualTape(m *FlowHeadCPU, condition, times, input []float32, index int, shaderNorm bool) *dualFlowTape {
	t := &dualFlowTape{times: make([]dualTimeTape, len(times)), blocks: make([]dualBlockTape, len(m.Blocks))}
	x := dualVector{value: append([]float32(nil), input...), tangent: make([]float32, len(input))}
	c := dualVector{value: append([]float32(nil), condition...), tangent: make([]float32, len(condition))}
	t.input = dualLinearTape{input: x, output: sharedSerialFlowLinear(m.Input, x)}
	t.condition = dualLinearTape{input: c, output: sharedSerialFlowLinear(m.Condition, c)}
	t.conditionSum = cloneDual(t.condition.output)
	for i, model := range m.Time {
		dTime := float32(0)
		if i == index {
			dTime = 1
		}
		time := dualTimeTape{time: times[i], dTime: dTime}
		half := len(model.Frequencies)
		time.embedding = dualVector{value: make([]float32, 2*half), tangent: make([]float32, 2*half)}
		for j, freq := range model.Frequencies {
			angle := times[i] * freq
			s, c := float32(math.Sin(float64(angle))), float32(math.Cos(float64(angle)))
			time.embedding.value[j], time.embedding.value[half+j] = c, s
			time.embedding.tangent[j], time.embedding.tangent[half+j] = -s*freq*dTime, c*freq*dTime
		}
		time.fc1Pre = sharedSerialFlowLinear(model.FC1, time.embedding)
		time.fc1 = siluForwardDual(time.fc1Pre)
		time.fc2Pre = sharedSerialFlowLinear(model.FC2, time.fc1)
		if shaderNorm {
			time.output, time.mean, time.dMean, time.variance, time.dVariance, time.scale, time.dScale = sharedFlowRoundedNorm(time.fc2Pre, model.RMSWeight, nil, model.RMSEpsilon, true)
		} else {
			time.output, time.mean, time.dMean, time.variance, time.dVariance, time.scale, time.dScale = varianceNormForwardDualStats(time.fc2Pre, model.RMSWeight, model.RMSEpsilon)
		}
		t.times[i] = time
		for j := range t.conditionSum.value {
			t.conditionSum.value[j] += time.output.value[j] / float32(len(times))
			t.conditionSum.tangent[j] += time.output.tangent[j] / float32(len(times))
		}
	}
	x = cloneDual(t.input.output)
	for i, model := range m.Blocks {
		b := dualBlockTape{input: cloneDual(x), condition: cloneDual(t.conditionSum)}
		b.condAct = siluForwardDual(t.conditionSum)
		b.mod = sharedSerialFlowLinear(model.Modulation, b.condAct)
		if shaderNorm {
			b.norm, _, _, _, _, _, _ = sharedFlowRoundedNorm(x, model.NormWeight, model.NormBias, model.Epsilon, false)
		} else {
			b.norm = layerNormForwardDual(x, model.NormWeight, model.NormBias, model.Epsilon)
		}
		d := len(x.value)
		b.modulated = dualVector{value: make([]float32, d), tangent: make([]float32, d)}
		for j := 0; j < d; j++ {
			scale := 1 + b.mod.value[d+j]
			b.modulated.value[j] = b.norm.value[j]*scale + b.mod.value[j]
			b.modulated.tangent[j] = b.norm.tangent[j]*scale + b.norm.value[j]*b.mod.tangent[d+j] + b.mod.tangent[j]
		}
		b.fc1Pre = sharedSerialFlowLinear(model.FC1, b.modulated)
		b.fc1 = siluForwardDual(b.fc1Pre)
		b.update = sharedSerialFlowLinear(model.FC2, b.fc1)
		b.output = dualVector{value: make([]float32, d), tangent: make([]float32, d)}
		for j := 0; j < d; j++ {
			gate := b.mod.value[2*d+j]
			b.output.value[j] = x.value[j] + gate*b.update.value[j]
			b.output.tangent[j] = x.tangent[j] + b.mod.tangent[2*d+j]*b.update.value[j] + gate*b.update.tangent[j]
		}
		t.blocks[i] = b
		x = cloneDual(b.output)
	}
	model := m.Final
	f := dualFinalTape{input: cloneDual(x), condition: cloneDual(t.conditionSum)}
	f.condAct = siluForwardDual(t.conditionSum)
	f.mod = sharedSerialFlowLinear(model.Modulation, f.condAct)
	if shaderNorm {
		f.norm, _, _, _, _, _, _ = sharedFlowRoundedNorm(x, nil, nil, model.Epsilon, false)
	} else {
		f.norm = layerNormForwardDual(x, nil, nil, model.Epsilon)
	}
	d := len(x.value)
	f.modulated = dualVector{value: make([]float32, d), tangent: make([]float32, d)}
	for j := 0; j < d; j++ {
		scale := 1 + f.mod.value[d+j]
		f.modulated.value[j] = f.norm.value[j]*scale + f.mod.value[j]
		f.modulated.tangent[j] = f.norm.tangent[j]*scale + f.norm.value[j]*f.mod.tangent[d+j] + f.mod.tangent[j]
	}
	f.output = sharedSerialFlowLinear(model.Linear, f.modulated)
	t.final = f
	return t
}

// Zero-tangent forward data supplies the ordinary tape while backward keeps
// the exact default ordinary routines. Normalisation stats are recomputed from
// the new forward inputs, not borrowed from expected control values.
func sharedOrdinaryFlowTape(m *FlowHeadCPU, t *dualFlowTape) *flowHeadTape {
	out := &flowHeadTape{input: linearTape{input: t.input.input.value}, condition: linearTape{input: t.condition.input.value}, hidden: t.input.output.value, condBase: t.condition.output.value, cond: t.conditionSum.value, times: make([]timeTape, len(t.times)), blocks: make([]blockTape, len(t.blocks))}
	for i, v := range t.times {
		out.times[i] = timeTape{time: v.time, embedding: v.embedding.value, fc1Pre: v.fc1Pre.value, fc1: v.fc1.value, fc2Pre: v.fc2Pre.value, fc2: v.output.value, mean: v.mean, variance: v.variance, scale: v.scale}
	}
	for i, v := range t.blocks {
		base, mean, variance, inv := layerNormBase(v.input.value, m.Blocks[i].Epsilon)
		out.blocks[i] = blockTape{x: v.input.value, condition: v.condition.value, condAct: v.condAct.value, mod: v.mod.value, normBase: base, normAffine: v.norm.value, modulated: v.modulated.value, fc1Pre: v.fc1Pre.value, fc1: v.fc1.value, update: v.update.value, mean: mean, variance: variance, inv: inv}
	}
	v := t.final
	norm, mean, variance, inv := layerNormBase(v.input.value, m.Final.Epsilon)
	out.final = finalTape{x: v.input.value, condition: v.condition.value, condAct: v.condAct.value, mod: v.mod.value, norm: norm, modulated: v.modulated.value, mean: mean, variance: variance, inv: inv}
	return out
}

func (b *sharedFlowForwardSensitivityBackend) EvaluateFlow(m *FlowHeadCPU, c, times, x []float32, index int, gv, gt []float32) (FlowDualTrainingResult, error) {
	if b == nil {
		return FlowDualTrainingResult{}, fmt.Errorf("nil flow forward observer")
	}
	if err := validateTrainableFlowHead(m, c, times, x, gv); err != nil {
		return FlowDualTrainingResult{}, err
	}
	if len(gt) != len(gv) || !finiteF32(gt) || index < -1 || index >= len(times) || (index == -1 && !zeroTrainingSeed(gt)) {
		return FlowDualTrainingResult{}, fmt.Errorf("invalid flow tangent/index")
	}
	var r FlowDualTrainingResult
	var g *FlowHeadGradients
	var err error
	if !b.serial && b.backwardPolicy != nil {
		g = newFlowHeadGradients(m)
		if index < 0 {
			var tape *flowHeadTape
			tape, r.Output, err = m.forwardTraining(c, times, x)
			if err != nil {
				return FlowDualTrainingResult{}, err
			}
			r.Tangent = make([]float32, len(gv))
			r.DCondition, r.DTimes, r.DInput, err = sharedPolicyFlowBackward(m, tape, gv, g, nil, b.backwardPolicy)
		} else {
			tape := m.forwardDualTape(c, times, x, index)
			r.Output, r.Tangent = tape.final.output.value, tape.final.output.tangent
			r.DCondition, r.DTimes, r.DInput = sharedPolicyFlowDualBackward(m, tape, dualAdjoint{value: gv, tangent: gt}, g, nil, b.backwardPolicy)
		}
	} else if !b.serial && b.normAdjoint {
		g = newFlowHeadGradients(m)
		if index < 0 {
			r.Output, g, r.DCondition, r.DTimes, r.DInput, err = m.ForwardBackward(c, times, x, gv)
			r.Tangent = make([]float32, len(gv))
		} else {
			tape := m.forwardDualTape(c, times, x, index)
			r.Output, r.Tangent = tape.final.output.value, tape.final.output.tangent
			r.DCondition, r.DTimes, r.DInput = sharedFlowNormBackward(m, tape, dualAdjoint{value: gv, tangent: gt}, g, nil, &b.normObservation)
		}
	} else if !b.serial && b.serialCondition {
		g = newFlowHeadGradients(m)
		if index < 0 {
			var tape *flowHeadTape
			tape, r.Output, err = m.forwardTraining(c, times, x)
			if err != nil {
				return FlowDualTrainingResult{}, err
			}
			r.Tangent = make([]float32, len(gv))
			r.DCondition, r.DTimes, r.DInput = sharedFlowConditionBackward(m, tape, nil, gv, gt, g, &b.condition)
		} else {
			tape := m.forwardDualTape(c, times, x, index)
			r.Output, r.Tangent = tape.final.output.value, tape.final.output.tangent
			r.DCondition, r.DTimes, r.DInput = sharedFlowConditionBackward(m, nil, tape, gv, gt, g, &b.condition)
		}
	} else if !b.serial {
		if index < 0 {
			r.Output, g, r.DCondition, r.DTimes, r.DInput, err = m.ForwardBackward(c, times, x, gv)
			r.Tangent = make([]float32, len(gv))
		} else {
			r.Output, r.Tangent, g, r.DCondition, r.DTimes, r.DInput, err = m.ForwardTimeJVPBackward(c, times, x, index, gv, gt)
		}
	} else {
		tape := sharedSerialFlowDualTape(m, c, times, x, index, b.shaderNorm)
		r.Output, r.Tangent = append([]float32(nil), tape.final.output.value...), append([]float32(nil), tape.final.output.tangent...)
		g = newFlowHeadGradients(m)
		if b.backwardPolicy != nil {
			if index < 0 {
				r.DCondition, r.DTimes, r.DInput, err = sharedPolicyFlowBackward(m, sharedOrdinaryFlowTape(m, tape), gv, g, nil, b.backwardPolicy)
			} else {
				r.DCondition, r.DTimes, r.DInput = sharedPolicyFlowDualBackward(m, tape, dualAdjoint{value: gv, tangent: gt}, g, nil, b.backwardPolicy)
			}
		} else if b.serialCondition {
			if index < 0 {
				r.DCondition, r.DTimes, r.DInput = sharedFlowConditionBackward(m, sharedOrdinaryFlowTape(m, tape), nil, gv, gt, g, &b.condition)
			} else {
				r.DCondition, r.DTimes, r.DInput = sharedFlowConditionBackward(m, nil, tape, gv, gt, g, &b.condition)
			}
		} else if index < 0 {
			r.DCondition, r.DTimes, r.DInput, err = m.backwardTraining(sharedOrdinaryFlowTape(m, tape), gv, g)
		} else if b.normAdjoint {
			r.DCondition, r.DTimes, r.DInput = sharedFlowNormBackward(m, tape, dualAdjoint{value: gv, tangent: gt}, g, nil, &b.normObservation)
		} else {
			r.DCondition, r.DTimes, r.DInput = m.backwardDualTape(tape, dualAdjoint{value: gv, tangent: gt}, g)
		}
		// Independently recomputed control inputs are observations only.
		var v, d []float32
		if index < 0 {
			_, v, err = m.forwardTraining(c, times, x)
			d = make([]float32, len(v))
		} else {
			control := m.forwardDualTape(c, times, x, index)
			v, d = control.final.output.value, control.final.output.tangent
		}
		if err != nil {
			return FlowDualTrainingResult{}, err
		}
		for i := range v {
			b.output.add(v[i], r.Output[i])
			b.tangent.add(d[i], r.Tangent[i])
		}
	}
	if err != nil {
		return FlowDualTrainingResult{}, err
	}
	r.Parameters = map[string][]float32{}
	addFlowHeadGradientParameters(r.Parameters, g)
	b.calls++
	if index < 0 {
		b.ordinary++
	} else {
		b.dual++
	}
	return r, nil
}

func TestPocketSharedFlowForwardSensitivity(t *testing.T) {
	m := tinyTrainableFlowHead()
	c, times, x := []float32{.2, -.3, .5}, []float32{.3, .8}, []float32{.3, -.2}
	gv, gt := []float32{.1, .2}, []float32{.3, -.4}
	for _, index := range []int{-1, 0, 1} {
		tangent := gt
		if index < 0 {
			tangent = []float32{0, 0}
		}
		control := &sharedFlowForwardSensitivityBackend{}
		want, err := control.EvaluateFlow(m, c, times, x, index, gv, tangent)
		if err != nil {
			t.Fatal(err)
		}
		b := &sharedFlowForwardSensitivityBackend{serial: true}
		r, err := b.EvaluateFlow(m, c, times, x, index, gv, tangent)
		if err != nil {
			t.Fatal(err)
		}
		if err := validateFlowBackendResult(m, r); err != nil {
			t.Fatal(err)
		}
		for i, pair := range [][2][]float32{{r.Output, want.Output}, {r.Tangent, want.Tangent}, {r.DInput, want.DInput}, {r.DCondition, want.DCondition}, {r.DTimes, want.DTimes}} {
			assertSliceClose(t, fmt.Sprint("serial flow boundary", i), pair[0], pair[1], 5e-5)
		}
		for name, v := range r.Parameters {
			assertSliceClose(t, name, v, want.Parameters[name], 5e-5)
		}
		if b.calls != 1 || b.output.Values != len(gv) || b.tangent.Values != len(gv) {
			t.Fatal("flow observation inventory")
		}
	}
	if _, err := (*sharedFlowForwardSensitivityBackend)(nil).EvaluateFlow(m, c, times, x, 0, gv, gt); err == nil {
		t.Fatal("nil observer")
	}
	for _, index := range []int{-2, 2} {
		if _, err := (&sharedFlowForwardSensitivityBackend{}).EvaluateFlow(m, c, times, x, index, gv, gt); err == nil {
			t.Fatal("invalid index")
		}
	}
	if _, err := (&sharedFlowForwardSensitivityBackend{}).EvaluateFlow(m, c, times, x, -1, gv, gt); err == nil {
		t.Fatal("ordinary tangent seed")
	}
	if _, err := (&sharedFlowForwardSensitivityBackend{}).EvaluateFlow(nil, c, times, x, 0, gv, gt); err == nil {
		t.Fatal("nil model")
	}
}
