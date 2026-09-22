package pockettts

import (
	"fmt"
	"math"
)

// LinearF32Gradient has the same row-major [Out,In] layout as LinearF32.
type LinearF32Gradient struct {
	Weight []float32
	Bias   []float32
}

type TimestepMLPGradient struct {
	FC1, FC2  LinearF32Gradient
	RMSWeight []float32
}

type AdaLNResidualGradient struct {
	NormWeight, NormBias []float32
	FC1, FC2             LinearF32Gradient
	Modulation           LinearF32Gradient
}

type AdaLNFinalGradient struct {
	Linear, Modulation LinearF32Gradient
}

// FlowHeadGradients mirrors every trainable parameter in FlowHeadCPU.
type FlowHeadGradients struct {
	Input, Condition LinearF32Gradient
	Time             []TimestepMLPGradient
	Blocks           []AdaLNResidualGradient
	Final            AdaLNFinalGradient
}

type linearTape struct {
	input []float32
}

type timeTape struct {
	time                   float32
	embedding, fc1Pre, fc1 []float32
	fc2Pre, fc2            []float32
	mean, variance, scale  float32
}

type blockTape struct {
	x, condition, condAct, mod, normBase, normAffine, modulated []float32
	fc1Pre, fc1, update                                         []float32
	mean, variance, inv                                         float32
}

type finalTape struct {
	x, condition, condAct, mod, norm, modulated []float32
	mean, variance, inv                         float32
}

type flowHeadTape struct {
	input, condition       linearTape
	hidden, condBase, cond []float32
	times                  []timeTape
	blocks                 []blockTape
	final                  finalTape
}

// ForwardBackward evaluates one F32-owned flow-head row and differentiates it.
// Inference BF16 weights are rejected: training owns mutable F32 parameters and
// must not silently update a decoded copy of an immutable checkpoint.
func (m *FlowHeadCPU) ForwardBackward(condition, times, input, dOutput []float32) (output []float32, gradients *FlowHeadGradients, dCondition, dTimes, dInput []float32, err error) {
	if err = validateTrainableFlowHead(m, condition, times, input, dOutput); err != nil {
		return nil, nil, nil, nil, nil, err
	}
	tape, output, err := m.forwardTraining(condition, times, input)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	gradients = newFlowHeadGradients(m)
	dCondition, dTimes, dInput, err = m.backwardTraining(tape, dOutput, gradients)
	if err != nil {
		return nil, nil, nil, nil, nil, err
	}
	return output, gradients, dCondition, dTimes, dInput, nil
}

func validateTrainableFlowHead(m *FlowHeadCPU, condition, times, input, dOutput []float32) error {
	if m == nil || len(m.Blocks) == 0 || len(times) != len(m.Time) || len(input) != m.Input.In || len(condition) != m.Condition.In || len(dOutput) != m.Final.Linear.Out {
		return fmt.Errorf("invalid Pocket TTS trainable flow-head input")
	}
	d := m.Input.Out
	if d < 2 || m.Condition.Out != d || m.Final.Linear.In != d || m.Final.Modulation.In != d || m.Final.Modulation.Out != 2*d || m.Final.Epsilon <= 0 {
		return fmt.Errorf("invalid Pocket TTS trainable flow-head topology")
	}
	if err := validateTrainableFlowLinear(m.Input); err != nil {
		return err
	}
	if err := validateTrainableFlowLinear(m.Condition); err != nil {
		return err
	}
	if err := validateTrainableFlowLinear(m.Final.Linear); err != nil {
		return err
	}
	if err := validateTrainableFlowLinear(m.Final.Modulation); err != nil {
		return err
	}
	for _, time := range m.Time {
		if err := validateTrainableFlowLinear(time.FC1); err != nil {
			return err
		}
		if err := validateTrainableFlowLinear(time.FC2); err != nil {
			return err
		}
		half := len(time.Frequencies)
		if half == 0 || time.FC1.In != 2*half || time.FC1.Out != d || time.FC2.In != d || time.FC2.Out != d || len(time.RMSWeight) != d || time.RMSEpsilon <= 0 {
			return fmt.Errorf("invalid Pocket TTS trainable timestep MLP")
		}
	}
	for _, block := range m.Blocks {
		if err := validateTrainableFlowLinear(block.FC1); err != nil {
			return err
		}
		if err := validateTrainableFlowLinear(block.FC2); err != nil {
			return err
		}
		if err := validateTrainableFlowLinear(block.Modulation); err != nil {
			return err
		}
		if len(block.NormWeight) != d || len(block.NormBias) != d || block.FC1.In != d || block.FC1.Out != d || block.FC2.In != d || block.FC2.Out != d || block.Modulation.In != d || block.Modulation.Out != 3*d || block.Epsilon <= 0 {
			return fmt.Errorf("invalid Pocket TTS trainable residual block")
		}
	}
	if !finiteF32(condition) || !finiteF32(times) || !finiteF32(input) || !finiteF32(dOutput) {
		return fmt.Errorf("Pocket TTS trainable flow-head input is non-finite")
	}
	return nil
}

func validateTrainableFlowLinear(linear LinearF32) error {
	if linear.In <= 0 || linear.Out <= 0 || len(linear.Weight) != linear.In*linear.Out || len(linear.WeightBF16) != 0 || (linear.Bias != nil && len(linear.Bias) != linear.Out) {
		return fmt.Errorf("Pocket TTS training requires owned F32 linear weights")
	}
	return nil
}

func newLinearGradient(linear LinearF32) LinearF32Gradient {
	gradient := LinearF32Gradient{Weight: make([]float32, len(linear.Weight))}
	if linear.Bias != nil {
		gradient.Bias = make([]float32, len(linear.Bias))
	}
	return gradient
}

func newFlowHeadGradients(m *FlowHeadCPU) *FlowHeadGradients {
	g := &FlowHeadGradients{
		Input:     newLinearGradient(m.Input),
		Condition: newLinearGradient(m.Condition),
		Time:      make([]TimestepMLPGradient, len(m.Time)),
		Blocks:    make([]AdaLNResidualGradient, len(m.Blocks)),
		Final: AdaLNFinalGradient{
			Linear:     newLinearGradient(m.Final.Linear),
			Modulation: newLinearGradient(m.Final.Modulation),
		},
	}
	for i, time := range m.Time {
		g.Time[i] = TimestepMLPGradient{FC1: newLinearGradient(time.FC1), FC2: newLinearGradient(time.FC2), RMSWeight: make([]float32, len(time.RMSWeight))}
	}
	for i, block := range m.Blocks {
		g.Blocks[i] = AdaLNResidualGradient{
			NormWeight: make([]float32, len(block.NormWeight)),
			NormBias:   make([]float32, len(block.NormBias)),
			FC1:        newLinearGradient(block.FC1),
			FC2:        newLinearGradient(block.FC2),
			Modulation: newLinearGradient(block.Modulation),
		}
	}
	return g
}

func (m *FlowHeadCPU) forwardTraining(condition, times, input []float32) (*flowHeadTape, []float32, error) {
	d := m.Input.Out
	tape := &flowHeadTape{
		input:     linearTape{input: append([]float32(nil), input...)},
		condition: linearTape{input: append([]float32(nil), condition...)},
		hidden:    linearForwardTraining(m.Input, input),
		condBase:  linearForwardTraining(m.Condition, condition),
		times:     make([]timeTape, len(m.Time)),
		blocks:    make([]blockTape, len(m.Blocks)),
	}
	tape.cond = append([]float32(nil), tape.condBase...)
	for i := range m.Time {
		tape.times[i] = timestepForwardTraining(m.Time[i], times[i])
		for j := 0; j < d; j++ {
			tape.cond[j] += tape.times[i].fc2[j] / float32(len(m.Time))
		}
	}
	x := append([]float32(nil), tape.hidden...)
	for i := range m.Blocks {
		tape.blocks[i], x = blockForwardTraining(m.Blocks[i], x, tape.cond)
	}
	tape.final, x = finalForwardTraining(m.Final, x, tape.cond)
	return tape, x, nil
}

func linearForwardTraining(linear LinearF32, input []float32) []float32 {
	output := make([]float32, linear.Out)
	for row := 0; row < linear.Out; row++ {
		value := float32(0)
		if linear.Bias != nil {
			value = linear.Bias[row]
		}
		for column, x := range input {
			value += linear.Weight[row*linear.In+column] * x
		}
		output[row] = value
	}
	return output
}

func timestepForwardTraining(model TimestepMLP, time float32) timeTape {
	half := len(model.Frequencies)
	tape := timeTape{time: time, embedding: make([]float32, 2*half)}
	for i, frequency := range model.Frequencies {
		angle := time * frequency
		tape.embedding[i] = float32(math.Cos(float64(angle)))
		tape.embedding[half+i] = float32(math.Sin(float64(angle)))
	}
	tape.fc1Pre = linearForwardTraining(model.FC1, tape.embedding)
	tape.fc1 = siluCopy(tape.fc1Pre)
	tape.fc2Pre = linearForwardTraining(model.FC2, tape.fc1)
	tape.mean, tape.variance, tape.scale = varianceNormStats(tape.fc2Pre, model.RMSEpsilon)
	tape.fc2 = make([]float32, len(tape.fc2Pre))
	for i := range tape.fc2 {
		tape.fc2[i] = tape.fc2Pre[i] * model.RMSWeight[i] * tape.scale
	}
	return tape
}

func blockForwardTraining(model AdaLNResidual, input, condition []float32) (blockTape, []float32) {
	tape := blockTape{x: append([]float32(nil), input...), condition: append([]float32(nil), condition...), condAct: siluCopy(condition)}
	tape.mod = linearForwardTraining(model.Modulation, tape.condAct)
	tape.normBase, tape.mean, tape.variance, tape.inv = layerNormBase(input, model.Epsilon)
	d := len(input)
	tape.normAffine = make([]float32, d)
	tape.modulated = make([]float32, d)
	for i := 0; i < d; i++ {
		tape.normAffine[i] = tape.normBase[i]*model.NormWeight[i] + model.NormBias[i]
		tape.modulated[i] = tape.normAffine[i]*(1+tape.mod[d+i]) + tape.mod[i]
	}
	tape.fc1Pre = linearForwardTraining(model.FC1, tape.modulated)
	tape.fc1 = siluCopy(tape.fc1Pre)
	tape.update = linearForwardTraining(model.FC2, tape.fc1)
	output := make([]float32, d)
	for i := range output {
		output[i] = input[i] + tape.mod[2*d+i]*tape.update[i]
	}
	return tape, output
}

func finalForwardTraining(model AdaLNFinal, input, condition []float32) (finalTape, []float32) {
	tape := finalTape{x: append([]float32(nil), input...), condition: append([]float32(nil), condition...), condAct: siluCopy(condition)}
	tape.mod = linearForwardTraining(model.Modulation, tape.condAct)
	tape.norm, tape.mean, tape.variance, tape.inv = layerNormBase(input, model.Epsilon)
	d := len(input)
	tape.modulated = make([]float32, d)
	for i := range input {
		tape.modulated[i] = tape.norm[i]*(1+tape.mod[d+i]) + tape.mod[i]
	}
	return tape, linearForwardTraining(model.Linear, tape.modulated)
}

func siluCopy(input []float32) []float32 {
	output := make([]float32, len(input))
	for i, x := range input {
		output[i] = x * sigmoidF32(x)
	}
	return output
}

func sigmoidF32(x float32) float32 {
	if x >= 0 {
		z := float32(math.Exp(float64(-x)))
		return 1 / (1 + z)
	}
	z := float32(math.Exp(float64(x)))
	return z / (1 + z)
}

func layerNormBase(input []float32, epsilon float32) (output []float32, mean, variance, inv float32) {
	for _, value := range input {
		mean += value
	}
	mean /= float32(len(input))
	for _, value := range input {
		difference := value - mean
		variance += difference * difference
	}
	variance /= float32(len(input))
	inv = float32(1 / math.Sqrt(float64(variance+epsilon)))
	output = make([]float32, len(input))
	for i, value := range input {
		output[i] = (value - mean) * inv
	}
	return output, mean, variance, inv
}

func varianceNormStats(input []float32, epsilon float32) (mean, variance, scale float32) {
	for _, value := range input {
		mean += value
	}
	mean /= float32(len(input))
	for _, value := range input {
		difference := value - mean
		variance += difference * difference
	}
	variance /= float32(len(input) - 1)
	scale = float32(1 / math.Sqrt(float64(variance+epsilon)))
	return mean, variance, scale
}

func (m *FlowHeadCPU) backwardTraining(tape *flowHeadTape, dOutput []float32, gradients *FlowHeadGradients) (dCondition, dTimes, dInput []float32, err error) {
	dHidden, dCond := finalBackwardTraining(m.Final, tape.final, dOutput, &gradients.Final)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		var blockCond []float32
		dHidden, blockCond = blockBackwardTraining(m.Blocks[i], tape.blocks[i], dHidden, &gradients.Blocks[i])
		addInPlace(dCond, blockCond)
	}
	dInput = linearBackwardTraining(m.Input, tape.input.input, dHidden, &gradients.Input)
	dTimes = make([]float32, len(m.Time))
	for i := range m.Time {
		dTimeOutput := make([]float32, len(dCond))
		for j := range dCond {
			dTimeOutput[j] = dCond[j] / float32(len(m.Time))
		}
		dTimes[i] = timestepBackwardTraining(m.Time[i], tape.times[i], dTimeOutput, &gradients.Time[i])
	}
	dCondition = linearBackwardTraining(m.Condition, tape.condition.input, dCond, &gradients.Condition)
	return dCondition, dTimes, dInput, nil
}

func linearBackwardTraining(linear LinearF32, input, dOutput []float32, gradient *LinearF32Gradient) []float32 {
	dInput := make([]float32, linear.In)
	for row, d := range dOutput {
		if gradient.Bias != nil {
			gradient.Bias[row] += d
		}
		for column, x := range input {
			gradient.Weight[row*linear.In+column] += d * x
			dInput[column] += d * linear.Weight[row*linear.In+column]
		}
	}
	return dInput
}

func finalBackwardTraining(model AdaLNFinal, tape finalTape, dOutput []float32, gradient *AdaLNFinalGradient) (dInput, dCondition []float32) {
	dModulated := linearBackwardTraining(model.Linear, tape.modulated, dOutput, &gradient.Linear)
	d := len(tape.x)
	dNorm := make([]float32, d)
	dMod := make([]float32, 2*d)
	for i, g := range dModulated {
		dNorm[i] = g * (1 + tape.mod[d+i])
		dMod[i] = g
		dMod[d+i] = g * tape.norm[i]
	}
	dInput = layerNormBackward(tape.norm, tape.inv, dNorm)
	dCondAct := linearBackwardTraining(model.Modulation, tape.condAct, dMod, &gradient.Modulation)
	dCondition = siluBackward(tape.condition, dCondAct)
	return dInput, dCondition
}

func blockBackwardTraining(model AdaLNResidual, tape blockTape, dOutput []float32, gradient *AdaLNResidualGradient) (dInput, dCondition []float32) {
	d := len(tape.x)
	dInput = append([]float32(nil), dOutput...)
	dMod := make([]float32, 3*d)
	dUpdate := make([]float32, d)
	for i, g := range dOutput {
		dMod[2*d+i] = g * tape.update[i]
		dUpdate[i] = g * tape.mod[2*d+i]
	}
	dFC1 := linearBackwardTraining(model.FC2, tape.fc1, dUpdate, &gradient.FC2)
	dFC1Pre := siluBackward(tape.fc1Pre, dFC1)
	dModulated := linearBackwardTraining(model.FC1, tape.modulated, dFC1Pre, &gradient.FC1)
	dNormAffine := make([]float32, d)
	for i, g := range dModulated {
		dNormAffine[i] = g * (1 + tape.mod[d+i])
		dMod[i] += g
		dMod[d+i] += g * tape.normAffine[i]
		gradient.NormWeight[i] += dNormAffine[i] * tape.normBase[i]
		gradient.NormBias[i] += dNormAffine[i]
	}
	dNormBase := make([]float32, d)
	for i := range dNormBase {
		dNormBase[i] = dNormAffine[i] * model.NormWeight[i]
	}
	addInPlace(dInput, layerNormBackward(tape.normBase, tape.inv, dNormBase))
	dCondAct := linearBackwardTraining(model.Modulation, tape.condAct, dMod, &gradient.Modulation)
	dCondition = siluBackward(tape.condition, dCondAct)
	return dInput, dCondition
}

func timestepBackwardTraining(model TimestepMLP, tape timeTape, dOutput []float32, gradient *TimestepMLPGradient) float32 {
	dFC2 := varianceNormBackward(tape, model.RMSWeight, dOutput, gradient.RMSWeight)
	dFC1 := linearBackwardTraining(model.FC2, tape.fc1, dFC2, &gradient.FC2)
	dFC1Pre := siluBackward(tape.fc1Pre, dFC1)
	dEmbedding := linearBackwardTraining(model.FC1, tape.embedding, dFC1Pre, &gradient.FC1)
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

func varianceNormBackward(tape timeTape, alpha, dOutput []float32, dAlpha []float32) []float32 {
	n := len(tape.fc2Pre)
	dInput := make([]float32, n)
	dScale := float32(0)
	for i, value := range tape.fc2Pre {
		dAlpha[i] += dOutput[i] * value * tape.scale
		dInput[i] = dOutput[i] * alpha[i] * tape.scale
		dScale += dOutput[i] * value * alpha[i]
	}
	dVariance := dScale * (-0.5) * tape.scale * tape.scale * tape.scale
	for i, value := range tape.fc2Pre {
		dInput[i] += dVariance * 2 * (value - tape.mean) / float32(n-1)
	}
	return dInput
}

func layerNormBackward(normalized []float32, inv float32, dNormalized []float32) []float32 {
	n := float32(len(normalized))
	sum, dot := float32(0), float32(0)
	for i, g := range dNormalized {
		sum += g
		dot += g * normalized[i]
	}
	dInput := make([]float32, len(normalized))
	for i, g := range dNormalized {
		dInput[i] = inv * (g - sum/n - normalized[i]*dot/n)
	}
	return dInput
}

func siluBackward(input, dOutput []float32) []float32 {
	dInput := make([]float32, len(input))
	for i, x := range input {
		sigmoid := sigmoidF32(x)
		dInput[i] = dOutput[i] * sigmoid * (1 + x*(1-sigmoid))
	}
	return dInput
}

func addInPlace(dst, src []float32) {
	for i, value := range src {
		dst[i] += value
	}
}
