package pockettts

import (
	"fmt"
	"math"
)

type dualAdjoint struct {
	value, tangent []float32
}

type dualLinearTape struct {
	input  dualVector
	output dualVector
}

type dualTimeTape struct {
	time, dTime            float32
	embedding, fc1Pre, fc1 dualVector
	fc2Pre, output         dualVector
	mean, dMean            float32
	variance, dVariance    float32
	scale, dScale          float32
}

type dualBlockTape struct {
	input, condition, condAct, mod dualVector
	norm, modulated                dualVector
	fc1Pre, fc1, update, output    dualVector
}

type dualFinalTape struct {
	input, condition, condAct, mod dualVector
	norm, modulated, output        dualVector
}

type dualFlowTape struct {
	input, condition dualLinearTape
	times            []dualTimeTape
	conditionSum     dualVector
	blocks           []dualBlockTape
	final            dualFinalTape
}

// ForwardTimeJVPBackward differentiates a scalar objective that supplies
// independent seeds for the ordinary output and its selected-time JVP. It is
// exact reverse-over-forward AD: no finite differences enter training.
func (m *FlowHeadCPU) ForwardTimeJVPBackward(condition, times, input []float32, timeIndex int, dOutput, dTangent []float32) (output, tangent []float32, gradients *FlowHeadGradients, dCondition, dTimes, dInput []float32, err error) {
	if m == nil || m.Final.Linear.Out < 0 {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("invalid Pocket TTS time-JVP backward flow head")
	}
	if len(dOutput) != m.Final.Linear.Out || len(dTangent) != m.Final.Linear.Out {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("invalid Pocket TTS time-JVP backward seed shape")
	}
	if err = validateTrainableFlowHead(m, condition, times, input, dOutput); err != nil {
		return nil, nil, nil, nil, nil, nil, err
	}
	if timeIndex < 0 || timeIndex >= len(times) {
		return nil, nil, nil, nil, nil, nil, fmt.Errorf("Pocket TTS time-JVP backward index %d out of range", timeIndex)
	}
	for _, values := range [][]float32{dOutput, dTangent} {
		for _, value := range values {
			if !isFinite(value) {
				return nil, nil, nil, nil, nil, nil, fmt.Errorf("Pocket TTS time-JVP backward seed is non-finite")
			}
		}
	}
	tape := m.forwardDualTape(condition, times, input, timeIndex)
	gradients = newFlowHeadGradients(m)
	dCondition, dTimes, dInput = m.backwardDualTape(tape, dualAdjoint{value: append([]float32(nil), dOutput...), tangent: append([]float32(nil), dTangent...)}, gradients)
	return append([]float32(nil), tape.final.output.value...), append([]float32(nil), tape.final.output.tangent...), gradients, dCondition, dTimes, dInput, nil
}

func (m *FlowHeadCPU) forwardDualTape(condition, times, input []float32, timeIndex int) *dualFlowTape {
	zeroInput := dualVector{value: append([]float32(nil), input...), tangent: make([]float32, len(input))}
	zeroCondition := dualVector{value: append([]float32(nil), condition...), tangent: make([]float32, len(condition))}
	tape := &dualFlowTape{times: make([]dualTimeTape, len(times)), blocks: make([]dualBlockTape, len(m.Blocks))}
	tape.input = dualLinearTape{input: zeroInput, output: linearForwardDual(m.Input, zeroInput)}
	tape.condition = dualLinearTape{input: zeroCondition, output: linearForwardDual(m.Condition, zeroCondition)}
	tape.conditionSum = cloneDual(tape.condition.output)
	for i := range times {
		dTime := float32(0)
		if i == timeIndex {
			dTime = 1
		}
		tape.times[i] = timestepForwardDualTape(m.Time[i], times[i], dTime)
		for j := range tape.conditionSum.value {
			tape.conditionSum.value[j] += tape.times[i].output.value[j] / float32(len(times))
			tape.conditionSum.tangent[j] += tape.times[i].output.tangent[j] / float32(len(times))
		}
	}
	x := cloneDual(tape.input.output)
	for i := range m.Blocks {
		tape.blocks[i] = blockForwardDualTape(m.Blocks[i], x, tape.conditionSum)
		x = cloneDual(tape.blocks[i].output)
	}
	tape.final = finalForwardDualTape(m.Final, x, tape.conditionSum)
	return tape
}

func cloneDual(input dualVector) dualVector {
	return dualVector{value: append([]float32(nil), input.value...), tangent: append([]float32(nil), input.tangent...)}
}

func timestepForwardDualTape(model TimestepMLP, time, dTime float32) dualTimeTape {
	half := len(model.Frequencies)
	tape := dualTimeTape{time: time, dTime: dTime}
	tape.embedding = dualVector{value: make([]float32, 2*half), tangent: make([]float32, 2*half)}
	for i, frequency := range model.Frequencies {
		angle := time * frequency
		sine, cosine := float32(math.Sin(float64(angle))), float32(math.Cos(float64(angle)))
		tape.embedding.value[i], tape.embedding.value[half+i] = cosine, sine
		tape.embedding.tangent[i] = -sine * frequency * dTime
		tape.embedding.tangent[half+i] = cosine * frequency * dTime
	}
	tape.fc1Pre = linearForwardDual(model.FC1, tape.embedding)
	tape.fc1 = siluForwardDual(tape.fc1Pre)
	tape.fc2Pre = linearForwardDual(model.FC2, tape.fc1)
	tape.output, tape.mean, tape.dMean, tape.variance, tape.dVariance, tape.scale, tape.dScale = varianceNormForwardDualStats(tape.fc2Pre, model.RMSWeight, model.RMSEpsilon)
	return tape
}

func blockForwardDualTape(model AdaLNResidual, input, condition dualVector) dualBlockTape {
	tape := dualBlockTape{input: cloneDual(input), condition: cloneDual(condition)}
	tape.condAct = siluForwardDual(condition)
	tape.mod = linearForwardDual(model.Modulation, tape.condAct)
	tape.norm = layerNormForwardDual(input, model.NormWeight, model.NormBias, model.Epsilon)
	d := len(input.value)
	tape.modulated = dualVector{value: make([]float32, d), tangent: make([]float32, d)}
	for i := 0; i < d; i++ {
		scale := 1 + tape.mod.value[d+i]
		tape.modulated.value[i] = tape.norm.value[i]*scale + tape.mod.value[i]
		tape.modulated.tangent[i] = tape.norm.tangent[i]*scale + tape.norm.value[i]*tape.mod.tangent[d+i] + tape.mod.tangent[i]
	}
	tape.fc1Pre = linearForwardDual(model.FC1, tape.modulated)
	tape.fc1 = siluForwardDual(tape.fc1Pre)
	tape.update = linearForwardDual(model.FC2, tape.fc1)
	tape.output = dualVector{value: make([]float32, d), tangent: make([]float32, d)}
	for i := 0; i < d; i++ {
		gate := tape.mod.value[2*d+i]
		tape.output.value[i] = input.value[i] + gate*tape.update.value[i]
		tape.output.tangent[i] = input.tangent[i] + tape.mod.tangent[2*d+i]*tape.update.value[i] + gate*tape.update.tangent[i]
	}
	return tape
}

func finalForwardDualTape(model AdaLNFinal, input, condition dualVector) dualFinalTape {
	tape := dualFinalTape{input: cloneDual(input), condition: cloneDual(condition)}
	tape.condAct = siluForwardDual(condition)
	tape.mod = linearForwardDual(model.Modulation, tape.condAct)
	tape.norm = layerNormForwardDual(input, nil, nil, model.Epsilon)
	d := len(input.value)
	tape.modulated = dualVector{value: make([]float32, d), tangent: make([]float32, d)}
	for i := 0; i < d; i++ {
		scale := 1 + tape.mod.value[d+i]
		tape.modulated.value[i] = tape.norm.value[i]*scale + tape.mod.value[i]
		tape.modulated.tangent[i] = tape.norm.tangent[i]*scale + tape.norm.value[i]*tape.mod.tangent[d+i] + tape.mod.tangent[i]
	}
	tape.output = linearForwardDual(model.Linear, tape.modulated)
	return tape
}

func (m *FlowHeadCPU) backwardDualTape(tape *dualFlowTape, seed dualAdjoint, gradients *FlowHeadGradients) (dCondition, dTimes, dInput []float32) {
	dHidden, dCond := finalBackwardDual(m.Final, tape.final, seed, &gradients.Final)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		var blockCond dualAdjoint
		dHidden, blockCond = blockBackwardDual(m.Blocks[i], tape.blocks[i], dHidden, &gradients.Blocks[i])
		addDualAdjoint(&dCond, blockCond)
	}
	dInputDual := linearBackwardDual(m.Input, tape.input, dHidden, &gradients.Input)
	dInput = dInputDual.value
	dTimes = make([]float32, len(m.Time))
	for i := range m.Time {
		seed := dualAdjoint{value: make([]float32, len(dCond.value)), tangent: make([]float32, len(dCond.tangent))}
		for j := range seed.value {
			seed.value[j] = dCond.value[j] / float32(len(m.Time))
			seed.tangent[j] = dCond.tangent[j] / float32(len(m.Time))
		}
		dTime, _ := timestepBackwardDual(m.Time[i], tape.times[i], seed, &gradients.Time[i])
		dTimes[i] = dTime
	}
	dConditionDual := linearBackwardDual(m.Condition, tape.condition, dCond, &gradients.Condition)
	return dConditionDual.value, dTimes, dInput
}

func linearBackwardDual(linear LinearF32, tape dualLinearTape, seed dualAdjoint, gradient *LinearF32Gradient) dualAdjoint {
	input := tape.input
	dInput := dualAdjoint{value: make([]float32, linear.In), tangent: make([]float32, linear.In)}
	for row := 0; row < linear.Out; row++ {
		if gradient.Bias != nil {
			gradient.Bias[row] += seed.value[row]
		}
		for column := 0; column < linear.In; column++ {
			weight := linear.Weight[row*linear.In+column]
			gradient.Weight[row*linear.In+column] += seed.value[row]*input.value[column] + seed.tangent[row]*input.tangent[column]
			dInput.value[column] += seed.value[row] * weight
			dInput.tangent[column] += seed.tangent[row] * weight
		}
	}
	return dInput
}

func finalBackwardDual(model AdaLNFinal, tape dualFinalTape, seed dualAdjoint, gradient *AdaLNFinalGradient) (dualAdjoint, dualAdjoint) {
	dModulated := linearBackwardDual(model.Linear, dualLinearTape{input: tape.modulated, output: tape.output}, seed, &gradient.Linear)
	d := len(tape.input.value)
	dNorm := zeroDualAdjoint(d)
	dMod := zeroDualAdjoint(2 * d)
	for i := 0; i < d; i++ {
		scale := 1 + tape.mod.value[d+i]
		dNorm.value[i] += dModulated.value[i]*scale + dModulated.tangent[i]*tape.mod.tangent[d+i]
		dNorm.tangent[i] += dModulated.tangent[i] * scale
		dMod.value[i] += dModulated.value[i]
		dMod.tangent[i] += dModulated.tangent[i]
		dMod.value[d+i] += dModulated.value[i]*tape.norm.value[i] + dModulated.tangent[i]*tape.norm.tangent[i]
		dMod.tangent[d+i] += dModulated.tangent[i] * tape.norm.value[i]
	}
	dInput := layerNormBackwardDual(tape.input, dNorm, nil, model.Epsilon, nil, nil)
	dCondAct := linearBackwardDual(model.Modulation, dualLinearTape{input: tape.condAct, output: tape.mod}, dMod, &gradient.Modulation)
	dCondition := siluBackwardDual(tape.condition, dCondAct)
	return dInput, dCondition
}

func blockBackwardDual(model AdaLNResidual, tape dualBlockTape, seed dualAdjoint, gradient *AdaLNResidualGradient) (dualAdjoint, dualAdjoint) {
	d := len(tape.input.value)
	dInput := dualAdjoint{value: append([]float32(nil), seed.value...), tangent: append([]float32(nil), seed.tangent...)}
	dMod := zeroDualAdjoint(3 * d)
	dUpdate := zeroDualAdjoint(d)
	for i := 0; i < d; i++ {
		gate := tape.mod.value[2*d+i]
		dMod.value[2*d+i] += seed.value[i]*tape.update.value[i] + seed.tangent[i]*tape.update.tangent[i]
		dMod.tangent[2*d+i] += seed.tangent[i] * tape.update.value[i]
		dUpdate.value[i] += seed.value[i]*gate + seed.tangent[i]*tape.mod.tangent[2*d+i]
		dUpdate.tangent[i] += seed.tangent[i] * gate
	}
	dFC1 := linearBackwardDual(model.FC2, dualLinearTape{input: tape.fc1, output: tape.update}, dUpdate, &gradient.FC2)
	dFC1Pre := siluBackwardDual(tape.fc1Pre, dFC1)
	dModulated := linearBackwardDual(model.FC1, dualLinearTape{input: tape.modulated, output: tape.fc1Pre}, dFC1Pre, &gradient.FC1)
	dNorm := zeroDualAdjoint(d)
	for i := 0; i < d; i++ {
		scale := 1 + tape.mod.value[d+i]
		dNorm.value[i] += dModulated.value[i]*scale + dModulated.tangent[i]*tape.mod.tangent[d+i]
		dNorm.tangent[i] += dModulated.tangent[i] * scale
		dMod.value[i] += dModulated.value[i]
		dMod.tangent[i] += dModulated.tangent[i]
		dMod.value[d+i] += dModulated.value[i]*tape.norm.value[i] + dModulated.tangent[i]*tape.norm.tangent[i]
		dMod.tangent[d+i] += dModulated.tangent[i] * tape.norm.value[i]
	}
	addDualAdjoint(&dInput, layerNormBackwardDual(tape.input, dNorm, model.NormWeight, model.Epsilon, gradient.NormWeight, gradient.NormBias))
	dCondAct := linearBackwardDual(model.Modulation, dualLinearTape{input: tape.condAct, output: tape.mod}, dMod, &gradient.Modulation)
	dCondition := siluBackwardDual(tape.condition, dCondAct)
	return dInput, dCondition
}

func timestepBackwardDual(model TimestepMLP, tape dualTimeTape, seed dualAdjoint, gradient *TimestepMLPGradient) (dTime, dDTime float32) {
	dFC2Pre := varianceNormBackwardDual(tape, model.RMSWeight, seed, gradient.RMSWeight)
	dFC1 := linearBackwardDual(model.FC2, dualLinearTape{input: tape.fc1, output: tape.fc2Pre}, dFC2Pre, &gradient.FC2)
	dFC1Pre := siluBackwardDual(tape.fc1Pre, dFC1)
	dEmbedding := linearBackwardDual(model.FC1, dualLinearTape{input: tape.embedding, output: tape.fc1Pre}, dFC1Pre, &gradient.FC1)
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

func siluBackwardDual(input dualVector, seed dualAdjoint) dualAdjoint {
	output := zeroDualAdjoint(len(input.value))
	for i, x := range input.value {
		s := sigmoidF32(x)
		first := s * (1 + x*(1-s))
		second := s * (1 - s) * (2 + x*(1-2*s))
		output.value[i] = seed.value[i]*first + seed.tangent[i]*second*input.tangent[i]
		output.tangent[i] = seed.tangent[i] * first
	}
	return output
}

func layerNormBackwardDual(input dualVector, seed dualAdjoint, weight []float32, epsilon float32, dWeight, dBias []float32) dualAdjoint {
	// Reverse the explicit dual forward with scalar reductions. This retains
	// population variance and works for affine and non-affine LayerNorm.
	n := len(input.value)
	nf := float32(n)
	mean, dMean := float32(0), float32(0)
	for i := range input.value {
		mean += input.value[i]
		dMean += input.tangent[i]
	}
	mean, dMean = mean/nf, dMean/nf
	center, dCenter := make([]float32, n), make([]float32, n)
	variance, dVariance := float32(0), float32(0)
	for i := range center {
		center[i], dCenter[i] = input.value[i]-mean, input.tangent[i]-dMean
		variance += center[i] * center[i]
		dVariance += 2 * center[i] * dCenter[i]
	}
	variance, dVariance = variance/nf, dVariance/nf
	inv := float32(1 / math.Sqrt(float64(variance+epsilon)))
	dInv := -0.5 * inv * inv * inv * dVariance
	gCenter, gDCenter := make([]float32, n), make([]float32, n)
	gInv, gDInv := float32(0), float32(0)
	for i := range center {
		base, tangentBase := center[i]*inv, dCenter[i]*inv+center[i]*dInv
		gv, gt := seed.value[i], seed.tangent[i]
		if weight != nil {
			if dWeight != nil {
				dWeight[i] += gv*base + gt*tangentBase
				dBias[i] += gv
			}
			gv, gt = gv*weight[i], gt*weight[i]
		}
		gCenter[i] += gv*inv + gt*dInv
		gDCenter[i] += gt * inv
		gInv += gv*center[i] + gt*dCenter[i]
		gDInv += gt * center[i]
	}
	gVariance := gInv*(-0.5)*inv*inv*inv + gDInv*(3.0/4.0)*inv*inv*inv*inv*inv*dVariance
	gDVariance := gDInv * (-0.5) * inv * inv * inv
	for i := range center {
		gCenter[i] += gVariance*2*center[i]/nf + gDVariance*2*dCenter[i]/nf
		gDCenter[i] += gDVariance * 2 * center[i] / nf
	}
	gMean, gDMean := float32(0), float32(0)
	for i := range center {
		gMean -= gCenter[i]
		gDMean -= gDCenter[i]
	}
	output := zeroDualAdjoint(n)
	for i := range output.value {
		output.value[i] = gCenter[i] + gMean/nf
		output.tangent[i] = gDCenter[i] + gDMean/nf
	}
	return output
}

func varianceNormBackwardDual(tape dualTimeTape, alpha []float32, seed dualAdjoint, dAlpha []float32) dualAdjoint {
	n := len(tape.fc2Pre.value)
	denom := float32(n - 1)
	gValue, gTangent := make([]float32, n), make([]float32, n)
	gScale, gDScale := float32(0), float32(0)
	for i := 0; i < n; i++ {
		v, dv := tape.fc2Pre.value[i], tape.fc2Pre.tangent[i]
		gv, gt := seed.value[i], seed.tangent[i]
		dAlpha[i] += gv*v*tape.scale + gt*(dv*tape.scale+v*tape.dScale)
		gValue[i] += gv*alpha[i]*tape.scale + gt*alpha[i]*tape.dScale
		gTangent[i] += gt * alpha[i] * tape.scale
		gScale += gv*alpha[i]*v + gt*alpha[i]*dv
		gDScale += gt * alpha[i] * v
	}
	gVariance := gScale*(-0.5)*tape.scale*tape.scale*tape.scale + gDScale*(3.0/4.0)*tape.scale*tape.scale*tape.scale*tape.scale*tape.scale*tape.dVariance
	gDVariance := gDScale * (-0.5) * tape.scale * tape.scale * tape.scale
	for i := 0; i < n; i++ {
		center := tape.fc2Pre.value[i] - tape.mean
		dCenter := tape.fc2Pre.tangent[i] - tape.dMean
		gValue[i] += gVariance*2*center/denom + gDVariance*2*dCenter/denom
		gTangent[i] += gDVariance * 2 * center / denom
	}
	// The centred variance contributions sum to zero over the channel axis;
	// the direct x*alpha*scale path is not centred and must not be included in
	// a mean-subtraction adjoint.
	return dualAdjoint{value: gValue, tangent: gTangent}
}

func varianceNormForwardDualStats(input dualVector, alpha []float32, epsilon float32) (output dualVector, mean, dMean, variance, dVariance, scale, dScale float32) {
	n := len(input.value)
	for i := range input.value {
		mean += input.value[i]
		dMean += input.tangent[i]
	}
	mean, dMean = mean/float32(n), dMean/float32(n)
	for i := range input.value {
		center, dCenter := input.value[i]-mean, input.tangent[i]-dMean
		variance += center * center
		dVariance += 2 * center * dCenter
	}
	variance, dVariance = variance/float32(n-1), dVariance/float32(n-1)
	scale = float32(1 / math.Sqrt(float64(variance+epsilon)))
	dScale = -0.5 * scale * scale * scale * dVariance
	output = dualVector{value: make([]float32, n), tangent: make([]float32, n)}
	for i := range input.value {
		output.value[i] = input.value[i] * alpha[i] * scale
		output.tangent[i] = alpha[i] * (input.tangent[i]*scale + input.value[i]*dScale)
	}
	return output, mean, dMean, variance, dVariance, scale, dScale
}

func zeroDualAdjoint(n int) dualAdjoint {
	return dualAdjoint{value: make([]float32, n), tangent: make([]float32, n)}
}

func addDualAdjoint(dst *dualAdjoint, src dualAdjoint) {
	for i := range dst.value {
		dst.value[i] += src.value[i]
		dst.tangent[i] += src.tangent[i]
	}
}
