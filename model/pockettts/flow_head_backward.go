package pockettts

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
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
	return m.forwardTrainingScratch(condition, times, input, nil)
}

func (m *FlowHeadCPU) forwardTrainingScratch(condition, times, input []float32, scratch *flowTapeScratch) (*flowHeadTape, []float32, error) {
	d := m.Input.Out
	tape := &flowHeadTape{
		input:     linearTape{input: scratch.copy(input)},
		condition: linearTape{input: scratch.copy(condition)},
		hidden:    linearForwardTrainingScratch(m.Input, input, scratch),
		condBase:  linearForwardTrainingScratch(m.Condition, condition, scratch),
		times:     make([]timeTape, len(m.Time)),
		blocks:    make([]blockTape, len(m.Blocks)),
	}
	tape.cond = scratch.copy(tape.condBase)
	for i := range m.Time {
		tape.times[i] = timestepForwardTrainingScratch(m.Time[i], times[i], scratch)
		for j := 0; j < d; j++ {
			tape.cond[j] += tape.times[i].fc2[j] / float32(len(m.Time))
		}
	}
	x := scratch.copy(tape.hidden)
	for i := range m.Blocks {
		tape.blocks[i], x = blockForwardTrainingScratch(m.Blocks[i], x, tape.cond, scratch)
	}
	tape.final, x = finalForwardTrainingScratch(m.Final, x, tape.cond, scratch)
	return tape, x, nil
}

func linearForwardTrainingInto(output []float32, linear LinearF32, input []float32) {
	if err := AffineSIMD(output, input, linear.Weight, linear.Bias, linear.In, linear.Out); err != nil {
		affineScalar(output, input, linear.Weight, linear.Bias, linear.In, linear.Out)
	}
}

func linearForwardTraining(linear LinearF32, input []float32) []float32 {
	return linearForwardTrainingScratch(linear, input, nil)
}

func linearForwardTrainingScratch(linear LinearF32, input []float32, scratch *flowTapeScratch) []float32 {
	output := scratch.take(linear.Out)
	linearForwardTrainingInto(output, linear, input)
	return output
}

func linearForwardRowsTraining(linear LinearF32, input []float32, rows int) []float32 {
	output := make([]float32, rows*linear.Out)
	if rows >= 16 {
		if err := linear.ForwardRows(output, input, rows); err == nil {
			return output
		}
	}
	for row := 0; row < rows; row++ {
		linearForwardTrainingInto(output[row*linear.Out:(row+1)*linear.Out], linear, input[row*linear.In:(row+1)*linear.In])
	}
	return output
}

func timestepForwardTraining(model TimestepMLP, time float32) timeTape {
	return timestepForwardTrainingScratch(model, time, nil)
}

func timestepForwardTrainingScratch(model TimestepMLP, time float32, scratch *flowTapeScratch) timeTape {
	half := len(model.Frequencies)
	tape := timeTape{time: time, embedding: scratch.take(2 * half)}
	for i, frequency := range model.Frequencies {
		angle := time * frequency
		tape.embedding[i] = float32(math.Cos(float64(angle)))
		tape.embedding[half+i] = float32(math.Sin(float64(angle)))
	}
	tape.fc1Pre = linearForwardTrainingScratch(model.FC1, tape.embedding, scratch)
	tape.fc1 = siluCopyScratch(tape.fc1Pre, scratch)
	tape.fc2Pre = linearForwardTrainingScratch(model.FC2, tape.fc1, scratch)
	tape.mean, tape.variance, tape.scale = varianceNormStats(tape.fc2Pre, model.RMSEpsilon)
	tape.fc2 = scratch.take(len(tape.fc2Pre))
	for i := range tape.fc2 {
		tape.fc2[i] = tape.fc2Pre[i] * model.RMSWeight[i] * tape.scale
	}
	return tape
}

func blockForwardTraining(model AdaLNResidual, input, condition []float32) (blockTape, []float32) {
	return blockForwardTrainingScratch(model, input, condition, nil)
}

func blockForwardTrainingScratch(model AdaLNResidual, input, condition []float32, scratch *flowTapeScratch) (blockTape, []float32) {
	tape := blockTape{x: scratch.copy(input), condition: scratch.copy(condition), condAct: siluCopyScratch(condition, scratch)}
	tape.mod = linearForwardTrainingScratch(model.Modulation, tape.condAct, scratch)
	tape.normBase, tape.mean, tape.variance, tape.inv = layerNormBaseScratch(input, model.Epsilon, scratch)
	d := len(input)
	tape.normAffine = scratch.take(d)
	tape.modulated = scratch.take(d)
	for i := 0; i < d; i++ {
		tape.normAffine[i] = tape.normBase[i]*model.NormWeight[i] + model.NormBias[i]
		tape.modulated[i] = tape.normAffine[i]*(1+tape.mod[d+i]) + tape.mod[i]
	}
	tape.fc1Pre = linearForwardTrainingScratch(model.FC1, tape.modulated, scratch)
	tape.fc1 = siluCopyScratch(tape.fc1Pre, scratch)
	tape.update = linearForwardTrainingScratch(model.FC2, tape.fc1, scratch)
	output := scratch.take(d)
	for i := range output {
		output[i] = input[i] + tape.mod[2*d+i]*tape.update[i]
	}
	return tape, output
}

func finalForwardTraining(model AdaLNFinal, input, condition []float32) (finalTape, []float32) {
	return finalForwardTrainingScratch(model, input, condition, nil)
}

func finalForwardTrainingScratch(model AdaLNFinal, input, condition []float32, scratch *flowTapeScratch) (finalTape, []float32) {
	tape := finalTape{x: scratch.copy(input), condition: scratch.copy(condition), condAct: siluCopyScratch(condition, scratch)}
	tape.mod = linearForwardTrainingScratch(model.Modulation, tape.condAct, scratch)
	tape.norm, tape.mean, tape.variance, tape.inv = layerNormBaseScratch(input, model.Epsilon, scratch)
	d := len(input)
	tape.modulated = scratch.take(d)
	for i := range input {
		tape.modulated[i] = tape.norm[i]*(1+tape.mod[d+i]) + tape.mod[i]
	}
	return tape, linearForwardTrainingScratch(model.Linear, tape.modulated, scratch)
}

func siluCopy(input []float32) []float32 {
	return siluCopyScratch(input, nil)
}

func siluCopyScratch(input []float32, scratch *flowTapeScratch) []float32 {
	output := scratch.take(len(input))
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
	return layerNormBaseScratch(input, epsilon, nil)
}

func layerNormBaseScratch(input []float32, epsilon float32, scratch *flowTapeScratch) (output []float32, mean, variance, inv float32) {
	output = scratch.take(len(input))
	mean, variance, inv = layerNormBaseInto(output, input, epsilon)
	return output, mean, variance, inv
}

func layerNormBaseInto(output, input []float32, epsilon float32) (mean, variance, inv float32) {
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
	for i, value := range input {
		output[i] = (value - mean) * inv
	}
	return mean, variance, inv
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
	return m.backwardTrainingScratch(tape, dOutput, gradients, nil)
}

func (m *FlowHeadCPU) backwardTrainingScratch(tape *flowHeadTape, dOutput []float32, gradients *FlowHeadGradients, scratch *flowTapeScratch) (dCondition, dTimes, dInput []float32, err error) {
	dHidden, dCond := finalBackwardTrainingScratch(m.Final, tape.final, dOutput, &gradients.Final, scratch)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		var blockCond []float32
		dHidden, blockCond = blockBackwardTrainingScratch(m.Blocks[i], tape.blocks[i], dHidden, &gradients.Blocks[i], scratch)
		addInPlace(dCond, blockCond)
	}
	dInput = linearBackwardTrainingScratch(m.Input, tape.input.input, dHidden, &gradients.Input, scratch)
	dTimes = scratch.take(len(m.Time))
	for i := range m.Time {
		dTimeOutput := scratch.take(len(dCond))
		for j := range dCond {
			dTimeOutput[j] = dCond[j] / float32(len(m.Time))
		}
		dTimes[i] = timestepBackwardTrainingScratch(m.Time[i], tape.times[i], dTimeOutput, &gradients.Time[i], scratch)
	}
	dCondition = linearBackwardTrainingScratch(m.Condition, tape.condition.input, dCond, &gradients.Condition, scratch)
	return dCondition, dTimes, dInput, nil
}

func linearBackwardTrainingInto(dInput []float32, linear LinearF32, input, dOutput []float32, gradient *LinearF32Gradient) {
	clear(dInput)
	if !simd.SgemmNNTo(dInput, dOutput, linear.Weight, 1, linear.In, linear.Out, 1, linear.Out, linear.In, linear.In) {
		for row, d := range dOutput {
			simd.VecScaleAdd(dInput, dInput, linear.Weight[row*linear.In:(row+1)*linear.In], d)
		}
	}
	for row, d := range dOutput {
		if gradient.Bias != nil {
			gradient.Bias[row] += d
		}
		weight := gradient.Weight[row*linear.In : (row+1)*linear.In]
		simd.VecScaleAdd(weight, weight, input, d)
	}
}

func linearBackwardTraining(linear LinearF32, input, dOutput []float32, gradient *LinearF32Gradient) []float32 {
	return linearBackwardTrainingScratch(linear, input, dOutput, gradient, nil)
}

func linearBackwardTrainingScratch(linear LinearF32, input, dOutput []float32, gradient *LinearF32Gradient, scratch *flowTapeScratch) []float32 {
	dInput := scratch.take(linear.In)
	linearBackwardTrainingInto(dInput, linear, input, dOutput, gradient)
	return dInput
}

func linearBackwardRowsTraining(linear LinearF32, input, dOutput []float32, rows int, gradient *LinearF32Gradient) []float32 {
	if rows <= 0 || len(input) != rows*linear.In || len(dOutput) != rows*linear.Out {
		return nil
	}
	dInput := make([]float32, rows*linear.In)
	if rows < 16 || !simd.DenseNNTo(dInput, dOutput, linear.Weight, rows, linear.In, linear.Out, 1, linear.Out, linear.In, linear.In) {
		for row := 0; row < rows; row++ {
			linearBackwardTrainingInto(dInput[row*linear.In:(row+1)*linear.In], linear, input[row*linear.In:(row+1)*linear.In], dOutput[row*linear.Out:(row+1)*linear.Out], gradient)
		}
		return dInput
	}
	transposed := make([]float32, linear.Out*rows)
	for row := 0; row < rows; row++ {
		for out := 0; out < linear.Out; out++ {
			transposed[out*rows+row] = dOutput[row*linear.Out+out]
		}
	}
	if !simd.DenseNNTo(gradient.Weight, transposed, input, linear.Out, linear.In, rows, 1, rows, linear.In, linear.In) {
		for row := 0; row < rows; row++ {
			for out, d := range dOutput[row*linear.Out : (row+1)*linear.Out] {
				weight := gradient.Weight[out*linear.In : (out+1)*linear.In]
				simd.VecScaleAdd(weight, weight, input[row*linear.In:(row+1)*linear.In], d)
			}
		}
	}
	if gradient.Bias != nil {
		for row := 0; row < rows; row++ {
			for out, d := range dOutput[row*linear.Out : (row+1)*linear.Out] {
				gradient.Bias[out] += d
			}
		}
	}
	return dInput
}

func finalBackwardTraining(model AdaLNFinal, tape finalTape, dOutput []float32, gradient *AdaLNFinalGradient) (dInput, dCondition []float32) {
	return finalBackwardTrainingScratch(model, tape, dOutput, gradient, nil)
}

func finalBackwardTrainingScratch(model AdaLNFinal, tape finalTape, dOutput []float32, gradient *AdaLNFinalGradient, scratch *flowTapeScratch) (dInput, dCondition []float32) {
	dModulated := linearBackwardTrainingScratch(model.Linear, tape.modulated, dOutput, &gradient.Linear, scratch)
	d := len(tape.x)
	dNorm := scratch.take(d)
	dMod := scratch.take(2 * d)
	for i, g := range dModulated {
		dNorm[i] = g * (1 + tape.mod[d+i])
		dMod[i] = g
		dMod[d+i] = g * tape.norm[i]
	}
	dInput = layerNormBackwardScratch(tape.norm, tape.inv, dNorm, scratch)
	dCondAct := linearBackwardTrainingScratch(model.Modulation, tape.condAct, dMod, &gradient.Modulation, scratch)
	dCondition = siluBackwardScratch(tape.condition, dCondAct, scratch)
	return dInput, dCondition
}

func blockBackwardTraining(model AdaLNResidual, tape blockTape, dOutput []float32, gradient *AdaLNResidualGradient) (dInput, dCondition []float32) {
	return blockBackwardTrainingScratch(model, tape, dOutput, gradient, nil)
}

func blockBackwardTrainingScratch(model AdaLNResidual, tape blockTape, dOutput []float32, gradient *AdaLNResidualGradient, scratch *flowTapeScratch) (dInput, dCondition []float32) {
	d := len(tape.x)
	dInput = scratch.copy(dOutput)
	dMod := scratch.take(3 * d)
	dUpdate := scratch.take(d)
	for i, g := range dOutput {
		dMod[2*d+i] = g * tape.update[i]
		dUpdate[i] = g * tape.mod[2*d+i]
	}
	dFC1 := linearBackwardTrainingScratch(model.FC2, tape.fc1, dUpdate, &gradient.FC2, scratch)
	dFC1Pre := siluBackwardScratch(tape.fc1Pre, dFC1, scratch)
	dModulated := linearBackwardTrainingScratch(model.FC1, tape.modulated, dFC1Pre, &gradient.FC1, scratch)
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
	dCondAct := linearBackwardTrainingScratch(model.Modulation, tape.condAct, dMod, &gradient.Modulation, scratch)
	dCondition = siluBackwardScratch(tape.condition, dCondAct, scratch)
	return dInput, dCondition
}

func timestepBackwardTraining(model TimestepMLP, tape timeTape, dOutput []float32, gradient *TimestepMLPGradient) float32 {
	return timestepBackwardTrainingScratch(model, tape, dOutput, gradient, nil)
}

func timestepBackwardTrainingScratch(model TimestepMLP, tape timeTape, dOutput []float32, gradient *TimestepMLPGradient, scratch *flowTapeScratch) float32 {
	dFC2 := varianceNormBackwardScratch(tape, model.RMSWeight, dOutput, gradient.RMSWeight, scratch)
	dFC1 := linearBackwardTrainingScratch(model.FC2, tape.fc1, dFC2, &gradient.FC2, scratch)
	dFC1Pre := siluBackwardScratch(tape.fc1Pre, dFC1, scratch)
	dEmbedding := linearBackwardTrainingScratch(model.FC1, tape.embedding, dFC1Pre, &gradient.FC1, scratch)
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
	return varianceNormBackwardScratch(tape, alpha, dOutput, dAlpha, nil)
}

func varianceNormBackwardScratch(tape timeTape, alpha, dOutput []float32, dAlpha []float32, scratch *flowTapeScratch) []float32 {
	n := len(tape.fc2Pre)
	dInput := scratch.take(n)
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
	return layerNormBackwardScratch(normalized, inv, dNormalized, nil)
}

func layerNormBackwardScratch(normalized []float32, inv float32, dNormalized []float32, scratch *flowTapeScratch) []float32 {
	dInput := scratch.take(len(normalized))
	layerNormBackwardInto(dInput, normalized, inv, dNormalized)
	return dInput
}

func layerNormBackwardInto(dInput, normalized []float32, inv float32, dNormalized []float32) {
	n := float32(len(normalized))
	sum, dot := float32(0), float32(0)
	for i, g := range dNormalized {
		sum += g
		dot += g * normalized[i]
	}
	for i, g := range dNormalized {
		dInput[i] = inv * (g - sum/n - normalized[i]*dot/n)
	}
}

func siluBackward(input, dOutput []float32) []float32 {
	return siluBackwardScratch(input, dOutput, nil)
}

func siluBackwardScratch(input, dOutput []float32, scratch *flowTapeScratch) []float32 {
	dInput := scratch.take(len(input))
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
