//go:build linux

package pockettts

import (
	"math"
	"testing"
)

// Source-formula F32 norm reverse-over-forward with CPU forward stats. Explicit
// test-only isolation: host arithmetic cannot establish native sqrt/exp parity.
type sharedFlowNormAdjointObservation struct {
	Calls                  int
	Value, Tangent, Weight sharedAttentionError
}

func sharedFlowNormAdjoint(input dualVector, seed dualAdjoint, weight []float32, epsilon float32, dWeight, dBias []float32, varianceOnly bool, stats *dualTimeTape, o *sharedFlowNormAdjointObservation) dualAdjoint {
	n := len(input.value)
	nf := float32(n)
	var mean, dm, variance, dv, inv, dinv float32
	if stats != nil {
		mean, dm, dv, inv, dinv = stats.mean, stats.dMean, stats.dVariance, stats.scale, stats.dScale
	} else {
		for i, x := range input.value {
			mean += x
			dm += input.tangent[i]
		}
		mean /= nf
		dm /= nf
		for i, x := range input.value {
			c, dc := x-mean, input.tangent[i]-dm
			variance += float32(c * c)
			dv += float32((2 * c) * dc)
		}
		variance /= nf
		dv /= nf
		inv = float32(1 / math.Sqrt(float64(variance+epsilon)))
		dinv = (((-.5 * inv) * inv) * inv) * dv
	}
	var ginv, gdinv float32
	for i, x := range input.value {
		v, t := x-mean, input.tangent[i]-dm
		if varianceOnly {
			v, t = x, input.tangent[i]
		}
		w := float32(1)
		if weight != nil {
			w = weight[i]
		}
		g, h := seed.value[i]*w, seed.tangent[i]*w
		ginv += float32(g*v + h*t)
		gdinv += float32(h * v)
		if dWeight != nil {
			tb := t*inv + v*dinv
			dWeight[i] += float32((seed.value[i]*v)*inv + seed.tangent[i]*tb)
			if dBias != nil {
				dBias[i] += seed.value[i]
			}
		}
	}
	gvar := (((ginv * (-.5)) * inv) * inv) * inv
	gvar += ((((((gdinv * .75) * inv) * inv) * inv) * inv) * inv) * dv
	gdvar := (((gdinv * (-.5)) * inv) * inv) * inv
	denom := nf
	if varianceOnly {
		denom = float32(n - 1)
	}
	out := zeroDualAdjoint(n)
	var total, ttotal float32
	for i, x := range input.value {
		w := float32(1)
		if weight != nil {
			w = weight[i]
		}
		g, h := seed.value[i]*w, seed.tangent[i]*w
		c, dc := x-mean, input.tangent[i]-dm
		v := g*inv + h*dinv
		v += ((gvar*2)*c)/denom + ((gdvar*2)*dc)/denom
		t := h * inv
		t += ((gdvar * 2) * c) / denom
		out.value[i], out.tangent[i] = v, t
		total -= v
		ttotal -= t
	}
	if !varianceOnly {
		for i := range out.value {
			out.value[i] += total / nf
			out.tangent[i] += ttotal / nf
		}
	}
	if o != nil {
		o.Calls++
	}
	return out
}

func sharedFlowNormAdjointObserved(input dualVector, seed dualAdjoint, weight []float32, epsilon float32, dWeight, dBias []float32, varianceOnly bool, stats *dualTimeTape, o *sharedFlowNormAdjointObservation) dualAdjoint {
	var cw, cb []float32
	if dWeight != nil {
		cw = make([]float32, len(dWeight))
		if dBias != nil {
			cb = make([]float32, len(dBias))
		}
	}
	var cpu dualAdjoint
	if varianceOnly {
		cpu = varianceNormBackwardDual(*stats, weight, seed, cw)
	} else {
		cpu = layerNormBackwardDual(input, seed, weight, epsilon, cw, cb)
	}
	before := append([]float32(nil), dWeight...)
	out := sharedFlowNormAdjoint(input, seed, weight, epsilon, dWeight, dBias, varianceOnly, stats, o)
	if o != nil {
		for i := range out.value {
			o.Value.add(cpu.value[i], out.value[i])
			o.Tangent.add(cpu.tangent[i], out.tangent[i])
		}
		for i := range dWeight {
			o.Weight.add(cw[i], dWeight[i]-before[i])
		}
	}
	return out
}

// Independent F64 norm/JVP map, used only by tiny finite-difference assertions.
func sharedNormDualObjective(x, xt, weight, gv, gt []float64, epsilon float64, varianceOnly bool) float64 {
	var mean, dm, v, dv float64
	n := float64(len(x))
	for i, a := range x {
		mean += a
		dm += xt[i]
	}
	mean /= n
	dm /= n
	denom := n
	if varianceOnly {
		denom = n - 1
	}
	for i, a := range x {
		c, dc := a-mean, xt[i]-dm
		v += c * c
		dv += 2 * c * dc
	}
	v /= denom
	dv /= denom
	inv := 1 / math.Sqrt(v+epsilon)
	dinv := -.5 * inv * inv * inv * dv
	var objective float64
	for i, a := range x {
		c, dc := a-mean, xt[i]-dm
		if varianceOnly {
			c, dc = a, xt[i]
		}
		objective += gv[i]*(c*weight[i]*inv) + gt[i]*weight[i]*(dc*inv+c*dinv)
	}
	return objective
}

func TestPocketSharedFlowNormAdjointFormula(t *testing.T) {
	input := dualVector{value: []float32{-.4, .2, .9, -.1}, tangent: []float32{.2, -.3, .1, .4}}
	seed := dualAdjoint{value: []float32{.1, -.2, .3, -.1}, tangent: []float32{-.2, .1, -.1, .3}}
	weight := []float32{.9, 1.1, .7, 1.2}
	to64 := func(a []float32) []float64 {
		out := make([]float64, len(a))
		for i, x := range a {
			out[i] = float64(x)
		}
		return out
	}
	for _, varianceOnly := range []bool{false, true} {
		dw, db := make([]float32, 4), make([]float32, 4)
		var stats *dualTimeTape
		if varianceOnly {
			s := dualTimeTape{fc2Pre: input}
			_, s.mean, s.dMean, s.variance, s.dVariance, s.scale, s.dScale = varianceNormForwardDualStats(input, weight, 1e-5)
			stats = &s
			db = nil
		}
		var obs sharedFlowNormAdjointObservation
		got := sharedFlowNormAdjointObserved(input, seed, weight, 1e-5, dw, db, varianceOnly, stats, &obs)
		x, xt, w, gv, gt := to64(input.value), to64(input.tangent), to64(weight), to64(seed.value), to64(seed.tangent)
		for _, group := range []struct {
			values     []float64
			derivative []float32
		}{{x, got.value}, {xt, got.tangent}, {w, dw}} {
			for i := range group.values {
				old := group.values[i]
				group.values[i] = old + 1e-5
				hi := sharedNormDualObjective(x, xt, w, gv, gt, 1e-5, varianceOnly)
				group.values[i] = old - 1e-5
				lo := sharedNormDualObjective(x, xt, w, gv, gt, 1e-5, varianceOnly)
				group.values[i] = old
				ref := (hi - lo) / 2e-5
				if math.Abs(float64(group.derivative[i])-ref) > 3e-6 {
					t.Fatal("independent F64 dual norm derivative", varianceOnly, i, group.derivative[i], ref)
				}
			}
		}
		if obs.Calls != 1 || obs.Value.Values != 4 {
			t.Fatal("norm observation count")
		}
	}
}

func TestPocketSharedFlowNormAdjointBoundary(t *testing.T) {
	m := tinyTrainableFlowHead()
	c, times, x := []float32{.2, -.3, .5}, []float32{.3, .8}, []float32{.3, -.2}
	gv, gt := []float32{.1, .2}, []float32{.3, -.4}
	for _, serial := range []bool{false, true} {
		for _, index := range []int{-1, 0, 1} {
			tangent := gt
			if index < 0 {
				tangent = []float32{0, 0}
			}
			want, err := (&sharedFlowForwardSensitivityBackend{serial: serial}).EvaluateFlow(m, c, times, x, index, gv, tangent)
			if err != nil {
				t.Fatal(err)
			}
			b := &sharedFlowForwardSensitivityBackend{serial: serial, normAdjoint: true}
			r, err := b.EvaluateFlow(m, c, times, x, index, gv, tangent)
			if err != nil {
				t.Fatal(err)
			}
			if err := validateFlowBackendResult(m, r); err != nil {
				t.Fatal(err)
			}
			for _, pair := range [][2][]float32{{r.Output, want.Output}, {r.Tangent, want.Tangent}, {r.DCondition, want.DCondition}, {r.DInput, want.DInput}, {r.DTimes, want.DTimes}} {
				assertSliceClose(t, "dual norm adjoint boundary", pair[0], pair[1], 5e-5)
			}
			for name, v := range r.Parameters {
				assertSliceClose(t, name, v, want.Parameters[name], 5e-5)
			}
			if index >= 0 && b.normObservation.Calls != len(m.Blocks)+len(m.Time)+1 {
				t.Fatal("dual norm calls", b.normObservation.Calls)
			}
		}
	}
}

// Test-only orchestration copied mechanically from the reviewed CPU routines;
// only the three norm adjoint call sites change. Production remains intact.
func sharedFlowNormFinalBackward(model AdaLNFinal, tape dualFinalTape, seed dualAdjoint, gradient *AdaLNFinalGradient, scratch *flowTapeScratch, obs *sharedFlowNormAdjointObservation) (dualAdjoint, dualAdjoint) {
	dModulated := linearBackwardDualScratch(model.Linear, dualLinearTape{input: tape.modulated, output: tape.output}, seed, &gradient.Linear, scratch)
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
	dInput := sharedFlowNormAdjointObserved(tape.input, dNorm, nil, model.Epsilon, nil, nil, false, nil, obs)
	dCondAct := linearBackwardDualScratch(model.Modulation, dualLinearTape{input: tape.condAct, output: tape.mod}, dMod, &gradient.Modulation, scratch)
	dCondition := siluBackwardDualScratch(tape.condition, dCondAct, scratch)
	return dInput, dCondition
}

func sharedFlowNormBlockBackward(model AdaLNResidual, tape dualBlockTape, seed dualAdjoint, gradient *AdaLNResidualGradient, scratch *flowTapeScratch, obs *sharedFlowNormAdjointObservation) (dualAdjoint, dualAdjoint) {
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
	dFC1 := linearBackwardDualScratch(model.FC2, dualLinearTape{input: tape.fc1, output: tape.update}, dUpdate, &gradient.FC2, scratch)
	dFC1Pre := siluBackwardDualScratch(tape.fc1Pre, dFC1, scratch)
	dModulated := linearBackwardDualScratch(model.FC1, dualLinearTape{input: tape.modulated, output: tape.fc1Pre}, dFC1Pre, &gradient.FC1, scratch)
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
	addDualAdjoint(&dInput, sharedFlowNormAdjointObserved(tape.input, dNorm, model.NormWeight, model.Epsilon, gradient.NormWeight, gradient.NormBias, false, nil, obs))
	dCondAct := linearBackwardDualScratch(model.Modulation, dualLinearTape{input: tape.condAct, output: tape.mod}, dMod, &gradient.Modulation, scratch)
	dCondition := siluBackwardDualScratch(tape.condition, dCondAct, scratch)
	return dInput, dCondition
}

func sharedFlowNormTimeBackward(model TimestepMLP, tape dualTimeTape, seed dualAdjoint, gradient *TimestepMLPGradient, scratch *flowTapeScratch, obs *sharedFlowNormAdjointObservation) (dTime, dDTime float32) {
	dFC2Pre := sharedFlowNormAdjointObserved(tape.fc2Pre, seed, model.RMSWeight, model.RMSEpsilon, gradient.RMSWeight, nil, true, &tape, obs)
	dFC1 := linearBackwardDualScratch(model.FC2, dualLinearTape{input: tape.fc1, output: tape.fc2Pre}, dFC2Pre, &gradient.FC2, scratch)
	dFC1Pre := siluBackwardDualScratch(tape.fc1Pre, dFC1, scratch)
	dEmbedding := linearBackwardDualScratch(model.FC1, dualLinearTape{input: tape.embedding, output: tape.fc1Pre}, dFC1Pre, &gradient.FC1, scratch)
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

func sharedFlowNormBackward(m *FlowHeadCPU, tape *dualFlowTape, seed dualAdjoint, gradients *FlowHeadGradients, scratch *flowTapeScratch, obs *sharedFlowNormAdjointObservation) (dCondition, dTimes, dInput []float32) {
	dHidden, dCond := sharedFlowNormFinalBackward(m.Final, tape.final, seed, &gradients.Final, scratch, obs)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		var blockCond dualAdjoint
		dHidden, blockCond = sharedFlowNormBlockBackward(m.Blocks[i], tape.blocks[i], dHidden, &gradients.Blocks[i], scratch, obs)
		addDualAdjoint(&dCond, blockCond)
	}
	dInputDual := linearBackwardDualScratch(m.Input, tape.input, dHidden, &gradients.Input, scratch)
	dInput = dInputDual.value
	dTimes = scratch.take(len(m.Time))
	for i := range m.Time {
		seed := dualAdjoint{value: scratch.take(len(dCond.value)), tangent: scratch.take(len(dCond.tangent))}
		for j := range seed.value {
			seed.value[j] = dCond.value[j] / float32(len(m.Time))
			seed.tangent[j] = dCond.tangent[j] / float32(len(m.Time))
		}
		dTime, _ := sharedFlowNormTimeBackward(m.Time[i], tape.times[i], seed, &gradients.Time[i], scratch, obs)
		dTimes[i] = dTime
	}
	dConditionDual := linearBackwardDualScratch(m.Condition, tape.condition, dCond, &gradients.Condition, scratch)
	return dConditionDual.value, dTimes, dInput
}
