//go:build linux

package pockettts

import (
	"math"
	"testing"
)

// Condition adjoint observation at the actual last flow projection input.
// Serial F32 is an explicit diagnostic intervention. F64 is observation only;
// neither a stored expected result nor a device implementation enters AD.
type sharedFlowConditionObservation struct {
	Calls                                          int
	CPUvsSerial, CPUvsReference, SerialvsReference sharedAttentionError
	MaxRoundingBoundRatio                          float64
}

func sharedFlowConditionAdjoint(l LinearF32, seed, cpu []float32, o *sharedFlowConditionObservation) []float32 {
	out := make([]float32, l.In)
	u := math.Ldexp(1, -24)
	gamma := float64(l.Out+2) * u / (1 - float64(l.Out+2)*u)
	for col := 0; col < l.In; col++ {
		var sum float32
		var ref, magnitude float64
		for row := 0; row < l.Out; row++ {
			sum += float32(seed[row] * l.Weight[row*l.In+col])
			product := float64(seed[row]) * float64(l.Weight[row*l.In+col])
			ref += product
			magnitude += math.Abs(product)
		}
		out[col] = sum
		if o != nil {
			o.CPUvsSerial.add(cpu[col], sum)
			o.CPUvsReference.add(cpu[col], float32(ref))
			o.SerialvsReference.add(sum, float32(ref))
			bound := gamma*magnitude + float64(l.Out+2)*math.Ldexp(1, -149)
			if bound > 0 {
				o.MaxRoundingBoundRatio = max(o.MaxRoundingBoundRatio, math.Abs(float64(sum)-ref)/bound)
			}
		}
	}
	if o != nil {
		o.Calls++
	}
	return out
}

// All upstream backward operations and parameter-gradient accumulation keep the
// existing CPU formula/order. Capture dCond before its final projection, then
// change only the condition input-adjoint reduction (not dWeight or dInput).
func sharedFlowConditionBackward(m *FlowHeadCPU, ordinary *flowHeadTape, dual *dualFlowTape, gv, gt []float32, gradients *FlowHeadGradients, o *sharedFlowConditionObservation) (dc, dt, dx []float32) {
	if ordinary != nil {
		dHidden, dCond := finalBackwardTraining(m.Final, ordinary.final, gv, &gradients.Final)
		for i := len(m.Blocks) - 1; i >= 0; i-- {
			var cond []float32
			dHidden, cond = blockBackwardTraining(m.Blocks[i], ordinary.blocks[i], dHidden, &gradients.Blocks[i])
			addInPlace(dCond, cond)
		}
		dx = linearBackwardTraining(m.Input, ordinary.input.input, dHidden, &gradients.Input)
		dt = make([]float32, len(m.Time))
		for i := range m.Time {
			seed := make([]float32, len(dCond))
			for j := range seed {
				seed[j] = dCond[j] / float32(len(m.Time))
			}
			dt[i] = timestepBackwardTraining(m.Time[i], ordinary.times[i], seed, &gradients.Time[i])
		}
		cpu := linearBackwardTraining(m.Condition, ordinary.condition.input, dCond, &gradients.Condition)
		dc = sharedFlowConditionAdjoint(m.Condition, dCond, cpu, o)
		return
	}
	dHidden, dCond := finalBackwardDual(m.Final, dual.final, dualAdjoint{value: gv, tangent: gt}, &gradients.Final)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		var cond dualAdjoint
		dHidden, cond = blockBackwardDual(m.Blocks[i], dual.blocks[i], dHidden, &gradients.Blocks[i])
		addDualAdjoint(&dCond, cond)
	}
	dx = linearBackwardDual(m.Input, dual.input, dHidden, &gradients.Input).value
	dt = make([]float32, len(m.Time))
	for i := range m.Time {
		seed := zeroDualAdjoint(len(dCond.value))
		for j := range seed.value {
			seed.value[j] = dCond.value[j] / float32(len(m.Time))
			seed.tangent[j] = dCond.tangent[j] / float32(len(m.Time))
		}
		dt[i], _ = timestepBackwardDual(m.Time[i], dual.times[i], seed, &gradients.Time[i])
	}
	cpu := linearBackwardDual(m.Condition, dual.condition, dCond, &gradients.Condition)
	dc = sharedFlowConditionAdjoint(m.Condition, dCond.value, cpu.value, o)
	return
}

func TestPocketSharedFlowConditionAdjoint(t *testing.T) {
	l := LinearF32{In: 2, Out: 3, Weight: []float32{.25, -.5, .75, .125, -.25, .5}}
	seed := []float32{.25, -.5, .75}
	cpu := make([]float32, 2)
	linearBackwardTrainingInto(cpu, l, []float32{.3, -.2}, seed, &LinearF32Gradient{Weight: make([]float32, 6)})
	var o sharedFlowConditionObservation
	got := sharedFlowConditionAdjoint(l, seed, cpu, &o)
	assertSliceClose(t, "independent dyadic condition adjoint", got, []float32{-.5, .1875}, 2e-7)
	if o.Calls != 1 || o.CPUvsSerial.Values != 2 || o.MaxRoundingBoundRatio > 1 {
		t.Fatal("condition observation inventory/bound", o)
	}
	m := tinyTrainableFlowHead()
	c, times, x := []float32{.2, -.3, .5}, []float32{.3, .8}, []float32{.3, -.2}
	gv, gt := []float32{.1, .2}, []float32{.3, -.4}
	for _, serial := range []bool{false, true} {
		for _, index := range []int{-1, 0, 1} {
			tangent := gt
			if index < 0 {
				tangent = []float32{0, 0}
			}
			control := &sharedFlowForwardSensitivityBackend{serial: serial}
			want, err := control.EvaluateFlow(m, c, times, x, index, gv, tangent)
			if err != nil {
				t.Fatal(err)
			}
			b := &sharedFlowForwardSensitivityBackend{serial: serial, serialCondition: true}
			r, err := b.EvaluateFlow(m, c, times, x, index, gv, tangent)
			if err != nil {
				t.Fatal(err)
			}
			if err := validateFlowBackendResult(m, r); err != nil {
				t.Fatal(err)
			}
			for _, pair := range [][2][]float32{{r.Output, want.Output}, {r.Tangent, want.Tangent}, {r.DInput, want.DInput}, {r.DTimes, want.DTimes}, {r.DCondition, want.DCondition}} {
				assertSliceClose(t, "condition isolation", pair[0], pair[1], 5e-5)
			}
			for name, v := range r.Parameters {
				assertSliceClose(t, name, v, want.Parameters[name], 5e-5)
			}
			if b.condition.Calls != 1 || b.condition.CPUvsSerial.Values != len(c) || b.condition.MaxRoundingBoundRatio > 1 {
				t.Fatal("condition boundary", b.condition)
			}
		}
	}
}
