//go:build linux

package pockettts

import (
	"math"
	"testing"
)

type sharedFlowBackwardPolicy struct {
	Linear, SiLU                          bool
	LinearCalls, SiLUCalls                int
	Input, Weight, SiLUValue, SiLUTangent sharedAttentionError
}

// Explicit source-formula reduction. Weight and bias contributions retain
// primal-before-tangent order; no implicit reference fallback exists.
func sharedFlowLinearBackward(l LinearF32, x, seed []float32, g *LinearF32Gradient, scratch *flowTapeScratch, p *sharedFlowBackwardPolicy) []float32 {
	if !p.Linear {
		return linearBackwardTrainingScratch(l, x, seed, g, scratch)
	}
	cg := newLinearGradient(l)
	cpu := linearBackwardTrainingScratch(l, x, seed, &cg, scratch)
	out := scratch.take(l.In)
	for col := 0; col < l.In; col++ {
		var sum float32
		for row := 0; row < l.Out; row++ {
			sum += float32(seed[row] * l.Weight[row*l.In+col])
		}
		out[col] = sum
		p.Input.add(cpu[col], sum)
	}
	for row, d := range seed {
		if g.Bias != nil {
			g.Bias[row] += d
		}
		for col, v := range x {
			at := row*l.In + col
			contribution := float32(d * v)
			g.Weight[at] += contribution
			// Sample fixed evenly spaced gradient entries for observation; all entries
			// still use the independent serial formula and participate in AD.
			if at%max(1, len(g.Weight)/256) == 0 {
				p.Weight.add(cg.Weight[at], contribution)
			}
		}
	}
	p.LinearCalls++
	return out
}
func sharedFlowLinearDualBackward(l LinearF32, tape dualLinearTape, seed dualAdjoint, g *LinearF32Gradient, scratch *flowTapeScratch, p *sharedFlowBackwardPolicy) dualAdjoint {
	v := sharedFlowLinearBackward(l, tape.input.value, seed.value, g, scratch, p)
	noBias := LinearF32Gradient{Weight: g.Weight}
	t := sharedFlowLinearBackward(l, tape.input.tangent, seed.tangent, &noBias, scratch, p)
	return dualAdjoint{value: v, tangent: t}
}
func sharedFlowSiLUBackward(input, seed []float32, scratch *flowTapeScratch, p *sharedFlowBackwardPolicy) []float32 {
	if !p.SiLU {
		return siluBackwardScratch(input, seed, scratch)
	}
	cpu := siluBackwardScratch(input, seed, scratch)
	out := scratch.take(len(input))
	for i, x := range input {
		s := sigmoidF32(x)
		first := s * (1 + x*(1-s))
		out[i] = seed[i] * first
		p.SiLUValue.add(cpu[i], out[i])
	}
	p.SiLUCalls++
	return out
}
func sharedFlowSiLUDualBackward(input dualVector, seed dualAdjoint, scratch *flowTapeScratch, p *sharedFlowBackwardPolicy) dualAdjoint {
	if !p.SiLU {
		return siluBackwardDualScratch(input, seed, scratch)
	}
	cpu := siluBackwardDualScratch(input, seed, scratch)
	out := zeroDualAdjointScratch(len(input.value), scratch)
	for i, x := range input.value {
		s := sigmoidF32(x)
		first := s * (1 + x*(1-s))
		second := (s * (1 - s)) * (2 + x*(1-2*s))
		out.value[i] = seed.value[i]*first + (seed.tangent[i]*second)*input.tangent[i]
		out.tangent[i] = seed.tangent[i] * first
		p.SiLUValue.add(cpu.value[i], out.value[i])
		p.SiLUTangent.add(cpu.tangent[i], out.tangent[i])
	}
	p.SiLUCalls++
	return out
}

func TestPocketSharedFlowBackwardFormula(t *testing.T) {
	l := LinearF32{In: 2, Out: 3, Weight: []float32{.25, -.5, .75, .125, -.25, .5}, Bias: []float32{0, 0, 0}}
	x, seed := []float32{.3, -.2}, []float32{.25, -.5, .75}
	g := newLinearGradient(l)
	p := &sharedFlowBackwardPolicy{Linear: true, SiLU: true}
	got := sharedFlowLinearBackward(l, x, seed, &g, nil, p)
	assertSliceClose(t, "dyadic input adjoint", got, []float32{-.5, .1875}, 2e-7)
	for row, d := range seed {
		for col, v := range x {
			if math.Abs(float64(g.Weight[row*l.In+col])-float64(d)*float64(v)) > 2e-7 {
				t.Fatal("F64 weight reference")
			}
		}
	}
	input := dualVector{value: []float32{-4, -.1, 0, .2, 5}, tangent: []float32{.2, -.3, .5, .1, -.2}}
	ds := dualAdjoint{value: []float32{.1, .2, -.2, .3, -.1}, tangent: []float32{-.1, .3, .2, -.1, .2}}
	out := sharedFlowSiLUDualBackward(input, ds, nil, p)
	for i, a := range input.value {
		x := float64(a)
		s := 1 / (1 + math.Exp(-x))
		first := s * (1 + x*(1-s))
		second := s * (1 - s) * (2 + x*(1-2*s))
		v := float64(ds.value[i])*first + float64(ds.tangent[i])*second*float64(input.tangent[i])
		tangent := float64(ds.tangent[i]) * first
		if math.Abs(float64(out.value[i])-v) > 3e-6 || math.Abs(float64(out.tangent[i])-tangent) > 3e-6 {
			t.Fatal("F64 SiLU adjoint")
		}
	}
}

func TestPocketSharedFlowBackwardBoundary(t *testing.T) {
	m := tinyTrainableFlowHead()
	c, times, x := []float32{.2, -.3, .5}, []float32{.3, .8}, []float32{.3, -.2}
	gv, gt := []float32{.1, .2}, []float32{.3, -.4}
	for _, serial := range []bool{false, true} {
		for _, flags := range [][2]bool{{false, false}, {true, false}, {false, true}, {true, true}} {
			for _, index := range []int{-1, 0, 1} {
				tangent := gt
				if index < 0 {
					tangent = []float32{0, 0}
				}
				want, err := (&sharedFlowForwardSensitivityBackend{serial: serial}).EvaluateFlow(m, c, times, x, index, gv, tangent)
				if err != nil {
					t.Fatal(err)
				}
				policy := &sharedFlowBackwardPolicy{Linear: flags[0], SiLU: flags[1]}
				b := &sharedFlowForwardSensitivityBackend{serial: serial, backwardPolicy: policy}
				r, err := b.EvaluateFlow(m, c, times, x, index, gv, tangent)
				if err != nil {
					t.Fatal(err)
				}
				if err := validateFlowBackendResult(m, r); err != nil {
					t.Fatal(err)
				}
				for _, pair := range [][2][]float32{{r.Output, want.Output}, {r.Tangent, want.Tangent}, {r.DInput, want.DInput}, {r.DCondition, want.DCondition}, {r.DTimes, want.DTimes}} {
					assertSliceClose(t, "flow backward isolation", pair[0], pair[1], 5e-5)
				}
				for name, v := range r.Parameters {
					assertSliceClose(t, name, v, want.Parameters[name], 5e-5)
				}
				if flags[0] && policy.LinearCalls == 0 || flags[1] && policy.SiLUCalls == 0 {
					t.Fatal("backward call inventory")
				}
			}
		}
	}
}

// Test-only orchestration copied mechanically from reviewed CPU routines.
// Only linear/SiLU adjoint call sites change; production functions intact.
func sharedPolicyFinalBackward(model AdaLNFinal, tape finalTape, dOutput []float32, gradient *AdaLNFinalGradient, scratch *flowTapeScratch, policy *sharedFlowBackwardPolicy) (dInput, dCondition []float32) {
	dModulated := sharedFlowLinearBackward(model.Linear, tape.modulated, dOutput, &gradient.Linear, scratch, policy)
	d := len(tape.x)
	dNorm := scratch.take(d)
	dMod := scratch.take(2 * d)
	for i, g := range dModulated {
		dNorm[i] = g * (1 + tape.mod[d+i])
		dMod[i] = g
		dMod[d+i] = g * tape.norm[i]
	}
	dInput = layerNormBackwardScratch(tape.norm, tape.inv, dNorm, scratch)
	dCondAct := sharedFlowLinearBackward(model.Modulation, tape.condAct, dMod, &gradient.Modulation, scratch, policy)
	dCondition = sharedFlowSiLUBackward(tape.condition, dCondAct, scratch, policy)
	return dInput, dCondition
}

func sharedPolicyBlockBackward(model AdaLNResidual, tape blockTape, dOutput []float32, gradient *AdaLNResidualGradient, scratch *flowTapeScratch, policy *sharedFlowBackwardPolicy) (dInput, dCondition []float32) {
	d := len(tape.x)
	dInput = scratch.copy(dOutput)
	dMod := scratch.take(3 * d)
	dUpdate := scratch.take(d)
	for i, g := range dOutput {
		dMod[2*d+i] = g * tape.update[i]
		dUpdate[i] = g * tape.mod[2*d+i]
	}
	dFC1 := sharedFlowLinearBackward(model.FC2, tape.fc1, dUpdate, &gradient.FC2, scratch, policy)
	dFC1Pre := sharedFlowSiLUBackward(tape.fc1Pre, dFC1, scratch, policy)
	dModulated := sharedFlowLinearBackward(model.FC1, tape.modulated, dFC1Pre, &gradient.FC1, scratch, policy)
	dNormAffine := scratch.take(d)
	for i, g := range dModulated {
		dNormAffine[i] = g * (1 + tape.mod[d+i])
		dMod[i] += g
		dMod[d+i] += g * tape.normAffine[i]
		gradient.NormWeight[i] += dNormAffine[i] * tape.normBase[i]
		gradient.NormBias[i] += dNormAffine[i]
	}
	dNormBase := scratch.take(d)
	for i := range dNormBase {
		dNormBase[i] = dNormAffine[i] * model.NormWeight[i]
	}
	addInPlace(dInput, layerNormBackwardScratch(tape.normBase, tape.inv, dNormBase, scratch))
	dCondAct := sharedFlowLinearBackward(model.Modulation, tape.condAct, dMod, &gradient.Modulation, scratch, policy)
	dCondition = sharedFlowSiLUBackward(tape.condition, dCondAct, scratch, policy)
	return dInput, dCondition
}

func sharedPolicyTimeBackward(model TimestepMLP, tape timeTape, dOutput []float32, gradient *TimestepMLPGradient, scratch *flowTapeScratch, policy *sharedFlowBackwardPolicy) float32 {
	dFC2 := varianceNormBackwardScratch(tape, model.RMSWeight, dOutput, gradient.RMSWeight, scratch)
	dFC1 := sharedFlowLinearBackward(model.FC2, tape.fc1, dFC2, &gradient.FC2, scratch, policy)
	dFC1Pre := sharedFlowSiLUBackward(tape.fc1Pre, dFC1, scratch, policy)
	dEmbedding := sharedFlowLinearBackward(model.FC1, tape.embedding, dFC1Pre, &gradient.FC1, scratch, policy)
	half := len(model.Frequencies)
	dTime := float32(0)
	for i, frequency := range model.Frequencies {
		angle := tape.time * frequency
		sine := float32(math.Sin(float64(angle)))
		cosine := float32(math.Cos(float64(angle)))
		dTime += (-dEmbedding[i]*sine + dEmbedding[half+i]*cosine) * frequency
	}
	return dTime
}

func sharedPolicyFlowBackward(m *FlowHeadCPU, tape *flowHeadTape, dOutput []float32, gradients *FlowHeadGradients, scratch *flowTapeScratch, policy *sharedFlowBackwardPolicy) (dCondition, dTimes, dInput []float32, err error) {
	dHidden, dCond := sharedPolicyFinalBackward(m.Final, tape.final, dOutput, &gradients.Final, scratch, policy)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		var blockCond []float32
		dHidden, blockCond = sharedPolicyBlockBackward(m.Blocks[i], tape.blocks[i], dHidden, &gradients.Blocks[i], scratch, policy)
		addInPlace(dCond, blockCond)
	}
	dInput = sharedFlowLinearBackward(m.Input, tape.input.input, dHidden, &gradients.Input, scratch, policy)
	dTimes = scratch.take(len(m.Time))
	for i := range m.Time {
		dTimeOutput := scratch.take(len(dCond))
		for j := range dCond {
			dTimeOutput[j] = dCond[j] / float32(len(m.Time))
		}
		dTimes[i] = sharedPolicyTimeBackward(m.Time[i], tape.times[i], dTimeOutput, &gradients.Time[i], scratch, policy)
	}
	dCondition = sharedFlowLinearBackward(m.Condition, tape.condition.input, dCond, &gradients.Condition, scratch, policy)
	return dCondition, dTimes, dInput, nil
}

func sharedPolicyFinalDualBackward(model AdaLNFinal, tape dualFinalTape, seed dualAdjoint, gradient *AdaLNFinalGradient, scratch *flowTapeScratch, policy *sharedFlowBackwardPolicy) (dualAdjoint, dualAdjoint) {
	dModulated := sharedFlowLinearDualBackward(model.Linear, dualLinearTape{input: tape.modulated, output: tape.output}, seed, &gradient.Linear, scratch, policy)
	d := len(tape.input.value)
	dNorm := zeroDualAdjointScratch(d, scratch)
	dMod := zeroDualAdjointScratch(2*d, scratch)
	for i := 0; i < d; i++ {
		scale := 1 + tape.mod.value[d+i]
		dNorm.value[i] += dModulated.value[i]*scale + dModulated.tangent[i]*tape.mod.tangent[d+i]
		dNorm.tangent[i] += dModulated.tangent[i] * scale
		dMod.value[i] += dModulated.value[i]
		dMod.tangent[i] += dModulated.tangent[i]
		dMod.value[d+i] += dModulated.value[i]*tape.norm.value[i] + dModulated.tangent[i]*tape.norm.tangent[i]
		dMod.tangent[d+i] += dModulated.tangent[i] * tape.norm.value[i]
	}
	dInput := layerNormBackwardDualScratch(tape.input, dNorm, nil, model.Epsilon, nil, nil, scratch)
	dCondAct := sharedFlowLinearDualBackward(model.Modulation, dualLinearTape{input: tape.condAct, output: tape.mod}, dMod, &gradient.Modulation, scratch, policy)
	dCondition := sharedFlowSiLUDualBackward(tape.condition, dCondAct, scratch, policy)
	return dInput, dCondition
}

func sharedPolicyBlockDualBackward(model AdaLNResidual, tape dualBlockTape, seed dualAdjoint, gradient *AdaLNResidualGradient, scratch *flowTapeScratch, policy *sharedFlowBackwardPolicy) (dualAdjoint, dualAdjoint) {
	d := len(tape.input.value)
	dInput := dualAdjoint{value: scratch.copy(seed.value), tangent: scratch.copy(seed.tangent)}
	dMod := zeroDualAdjointScratch(3*d, scratch)
	dUpdate := zeroDualAdjointScratch(d, scratch)
	for i := 0; i < d; i++ {
		gate := tape.mod.value[2*d+i]
		dMod.value[2*d+i] += seed.value[i]*tape.update.value[i] + seed.tangent[i]*tape.update.tangent[i]
		dMod.tangent[2*d+i] += seed.tangent[i] * tape.update.value[i]
		dUpdate.value[i] += seed.value[i]*gate + seed.tangent[i]*tape.mod.tangent[2*d+i]
		dUpdate.tangent[i] += seed.tangent[i] * gate
	}
	dFC1 := sharedFlowLinearDualBackward(model.FC2, dualLinearTape{input: tape.fc1, output: tape.update}, dUpdate, &gradient.FC2, scratch, policy)
	dFC1Pre := sharedFlowSiLUDualBackward(tape.fc1Pre, dFC1, scratch, policy)
	dModulated := sharedFlowLinearDualBackward(model.FC1, dualLinearTape{input: tape.modulated, output: tape.fc1Pre}, dFC1Pre, &gradient.FC1, scratch, policy)
	dNorm := zeroDualAdjointScratch(d, scratch)
	for i := 0; i < d; i++ {
		scale := 1 + tape.mod.value[d+i]
		dNorm.value[i] += dModulated.value[i]*scale + dModulated.tangent[i]*tape.mod.tangent[d+i]
		dNorm.tangent[i] += dModulated.tangent[i] * scale
		dMod.value[i] += dModulated.value[i]
		dMod.tangent[i] += dModulated.tangent[i]
		dMod.value[d+i] += dModulated.value[i]*tape.norm.value[i] + dModulated.tangent[i]*tape.norm.tangent[i]
		dMod.tangent[d+i] += dModulated.tangent[i] * tape.norm.value[i]
	}
	addDualAdjoint(&dInput, layerNormBackwardDualScratch(tape.input, dNorm, model.NormWeight, model.Epsilon, gradient.NormWeight, gradient.NormBias, scratch))
	dCondAct := sharedFlowLinearDualBackward(model.Modulation, dualLinearTape{input: tape.condAct, output: tape.mod}, dMod, &gradient.Modulation, scratch, policy)
	dCondition := sharedFlowSiLUDualBackward(tape.condition, dCondAct, scratch, policy)
	return dInput, dCondition
}

func sharedPolicyTimeDualBackward(model TimestepMLP, tape dualTimeTape, seed dualAdjoint, gradient *TimestepMLPGradient, scratch *flowTapeScratch, policy *sharedFlowBackwardPolicy) (dTime, dDTime float32) {
	dFC2Pre := varianceNormBackwardDualScratch(tape, model.RMSWeight, seed, gradient.RMSWeight, scratch)
	dFC1 := sharedFlowLinearDualBackward(model.FC2, dualLinearTape{input: tape.fc1, output: tape.fc2Pre}, dFC2Pre, &gradient.FC2, scratch, policy)
	dFC1Pre := sharedFlowSiLUDualBackward(tape.fc1Pre, dFC1, scratch, policy)
	dEmbedding := sharedFlowLinearDualBackward(model.FC1, dualLinearTape{input: tape.embedding, output: tape.fc1Pre}, dFC1Pre, &gradient.FC1, scratch, policy)
	half := len(model.Frequencies)
	for i, frequency := range model.Frequencies {
		angle := tape.time * frequency
		sine, cosine := float32(math.Sin(float64(angle))), float32(math.Cos(float64(angle)))
		firstCos := -sine * frequency
		firstSin := cosine * frequency
		secondCos := -cosine * frequency * frequency
		secondSin := -sine * frequency * frequency
		dTime += dEmbedding.value[i]*firstCos + dEmbedding.value[half+i]*firstSin
		dTime += dEmbedding.tangent[i]*secondCos*tape.dTime + dEmbedding.tangent[half+i]*secondSin*tape.dTime
		dDTime += dEmbedding.tangent[i]*firstCos + dEmbedding.tangent[half+i]*firstSin
	}
	return dTime, dDTime
}

func sharedPolicyFlowDualBackward(m *FlowHeadCPU, tape *dualFlowTape, seed dualAdjoint, gradients *FlowHeadGradients, scratch *flowTapeScratch, policy *sharedFlowBackwardPolicy) (dCondition, dTimes, dInput []float32) {
	dHidden, dCond := sharedPolicyFinalDualBackward(m.Final, tape.final, seed, &gradients.Final, scratch, policy)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		var blockCond dualAdjoint
		dHidden, blockCond = sharedPolicyBlockDualBackward(m.Blocks[i], tape.blocks[i], dHidden, &gradients.Blocks[i], scratch, policy)
		addDualAdjoint(&dCond, blockCond)
	}
	dInputDual := sharedFlowLinearDualBackward(m.Input, tape.input, dHidden, &gradients.Input, scratch, policy)
	dInput = dInputDual.value
	dTimes = scratch.take(len(m.Time))
	for i := range m.Time {
		seed := dualAdjoint{value: scratch.take(len(dCond.value)), tangent: scratch.take(len(dCond.tangent))}
		for j := range seed.value {
			seed.value[j] = dCond.value[j] / float32(len(m.Time))
			seed.tangent[j] = dCond.tangent[j] / float32(len(m.Time))
		}
		dTime, _ := sharedPolicyTimeDualBackward(m.Time[i], tape.times[i], seed, &gradients.Time[i], scratch, policy)
		dTimes[i] = dTime
	}
	dConditionDual := sharedFlowLinearDualBackward(m.Condition, tape.condition, dCond, &gradients.Condition, scratch, policy)
	return dConditionDual.value, dTimes, dInput
}
