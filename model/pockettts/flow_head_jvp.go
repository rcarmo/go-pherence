package pockettts

import (
	"fmt"
	"math"
)

type dualVector struct {
	value, tangent []float32
}

// ForwardTimeJVP returns the flow-head output and its exact directional
// derivative for one scalar time condition. timeIndex selects which time has
// unit tangent; all parameters, the latent input and condition have zero
// tangent. The path requires owned F32 weights, as does ForwardBackward.
func (m *FlowHeadCPU) ForwardTimeJVP(condition, times, input []float32, timeIndex int) (output, tangent []float32, err error) {
	if m == nil || m.Final.Linear.Out < 0 {
		return nil, nil, fmt.Errorf("invalid Pocket TTS time JVP flow head")
	}
	dOutput := make([]float32, m.Final.Linear.Out)
	if err = validateTrainableFlowHead(m, condition, times, input, dOutput); err != nil {
		return nil, nil, err
	}
	if timeIndex < 0 || timeIndex >= len(times) {
		return nil, nil, fmt.Errorf("Pocket TTS time JVP index %d out of range", timeIndex)
	}
	x := linearForwardDual(m.Input, dualVector{value: input, tangent: make([]float32, len(input))})
	conditionDual := linearForwardDual(m.Condition, dualVector{value: condition, tangent: make([]float32, len(condition))})
	for i := range m.Time {
		tangent := float32(0)
		if i == timeIndex {
			tangent = 1
		}
		timeDual := timestepForwardDual(m.Time[i], times[i], tangent)
		for j := range conditionDual.value {
			conditionDual.value[j] += timeDual.value[j] / float32(len(m.Time))
			conditionDual.tangent[j] += timeDual.tangent[j] / float32(len(m.Time))
		}
	}
	for i := range m.Blocks {
		x = blockForwardDual(m.Blocks[i], x, conditionDual)
	}
	x = finalForwardDual(m.Final, x, conditionDual)
	return x.value, x.tangent, nil
}

func linearForwardDual(linear LinearF32, input dualVector) dualVector {
	return linearForwardDualScratch(linear, input, nil)
}

func linearForwardDualScratch(linear LinearF32, input dualVector, scratch *flowTapeScratch) dualVector {
	return dualVector{value: linearForwardTrainingScratch(linear, input.value, scratch), tangent: linearForwardTrainingScratch(LinearF32{Weight: linear.Weight, In: linear.In, Out: linear.Out}, input.tangent, scratch)}
}

func timestepForwardDual(model TimestepMLP, time, dTime float32) dualVector {
	half := len(model.Frequencies)
	embedding := dualVector{value: make([]float32, 2*half), tangent: make([]float32, 2*half)}
	for i, frequency := range model.Frequencies {
		angle := time * frequency
		sine := float32(math.Sin(float64(angle)))
		cosine := float32(math.Cos(float64(angle)))
		embedding.value[i] = cosine
		embedding.value[half+i] = sine
		embedding.tangent[i] = -sine * frequency * dTime
		embedding.tangent[half+i] = cosine * frequency * dTime
	}
	hidden := siluForwardDual(linearForwardDual(model.FC1, embedding))
	return varianceNormForwardDual(linearForwardDual(model.FC2, hidden), model.RMSWeight, model.RMSEpsilon)
}

func blockForwardDual(model AdaLNResidual, input, condition dualVector) dualVector {
	mod := linearForwardDual(model.Modulation, siluForwardDual(condition))
	norm := layerNormForwardDual(input, model.NormWeight, model.NormBias, model.Epsilon)
	d := len(input.value)
	modulated := dualVector{value: make([]float32, d), tangent: make([]float32, d)}
	for i := 0; i < d; i++ {
		scale := 1 + mod.value[d+i]
		modulated.value[i] = norm.value[i]*scale + mod.value[i]
		modulated.tangent[i] = norm.tangent[i]*scale + norm.value[i]*mod.tangent[d+i] + mod.tangent[i]
	}
	update := linearForwardDual(model.FC2, siluForwardDual(linearForwardDual(model.FC1, modulated)))
	output := dualVector{value: make([]float32, d), tangent: make([]float32, d)}
	for i := 0; i < d; i++ {
		gate := mod.value[2*d+i]
		output.value[i] = input.value[i] + gate*update.value[i]
		output.tangent[i] = input.tangent[i] + mod.tangent[2*d+i]*update.value[i] + gate*update.tangent[i]
	}
	return output
}

func finalForwardDual(model AdaLNFinal, input, condition dualVector) dualVector {
	mod := linearForwardDual(model.Modulation, siluForwardDual(condition))
	norm := layerNormForwardDual(input, nil, nil, model.Epsilon)
	d := len(input.value)
	modulated := dualVector{value: make([]float32, d), tangent: make([]float32, d)}
	for i := 0; i < d; i++ {
		scale := 1 + mod.value[d+i]
		modulated.value[i] = norm.value[i]*scale + mod.value[i]
		modulated.tangent[i] = norm.tangent[i]*scale + norm.value[i]*mod.tangent[d+i] + mod.tangent[i]
	}
	return linearForwardDual(model.Linear, modulated)
}

func siluForwardDual(input dualVector) dualVector {
	return siluForwardDualScratch(input, nil)
}

func siluForwardDualScratch(input dualVector, scratch *flowTapeScratch) dualVector {
	output := dualVector{value: scratch.take(len(input.value)), tangent: scratch.take(len(input.value))}
	for i, value := range input.value {
		sigmoid := sigmoidF32(value)
		output.value[i] = value * sigmoid
		output.tangent[i] = input.tangent[i] * sigmoid * (1 + value*(1-sigmoid))
	}
	return output
}

func layerNormForwardDual(input dualVector, weight, bias []float32, epsilon float32) dualVector {
	return layerNormForwardDualScratch(input, weight, bias, epsilon, nil)
}

func layerNormForwardDualScratch(input dualVector, weight, bias []float32, epsilon float32, scratch *flowTapeScratch) dualVector {
	n := float32(len(input.value))
	mean, dMean := float32(0), float32(0)
	for i, value := range input.value {
		mean += value
		dMean += input.tangent[i]
	}
	mean /= n
	dMean /= n
	variance, dVariance := float32(0), float32(0)
	for i, value := range input.value {
		center, dCenter := value-mean, input.tangent[i]-dMean
		variance += center * center
		dVariance += 2 * center * dCenter
	}
	variance /= n
	dVariance /= n
	inv := float32(1 / math.Sqrt(float64(variance+epsilon)))
	dInv := -0.5 * inv * inv * inv * dVariance
	output := dualVector{value: scratch.take(len(input.value)), tangent: scratch.take(len(input.value))}
	for i, value := range input.value {
		center, dCenter := value-mean, input.tangent[i]-dMean
		output.value[i] = center * inv
		output.tangent[i] = dCenter*inv + center*dInv
		if weight != nil {
			output.value[i] = output.value[i]*weight[i] + bias[i]
			output.tangent[i] *= weight[i]
		}
	}
	return output
}

func varianceNormForwardDual(input dualVector, alpha []float32, epsilon float32) dualVector {
	n := len(input.value)
	mean, dMean := float32(0), float32(0)
	for i, value := range input.value {
		mean += value
		dMean += input.tangent[i]
	}
	mean /= float32(n)
	dMean /= float32(n)
	variance, dVariance := float32(0), float32(0)
	for i, value := range input.value {
		center, dCenter := value-mean, input.tangent[i]-dMean
		variance += center * center
		dVariance += 2 * center * dCenter
	}
	variance /= float32(n - 1)
	dVariance /= float32(n - 1)
	scale := float32(1 / math.Sqrt(float64(variance+epsilon)))
	dScale := -0.5 * scale * scale * scale * dVariance
	output := dualVector{value: make([]float32, n), tangent: make([]float32, n)}
	for i, value := range input.value {
		output.value[i] = value * alpha[i] * scale
		output.tangent[i] = alpha[i] * (input.tangent[i]*scale + value*dScale)
	}
	return output
}
