package pockettts

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// TrainingStepSamples freezes all stochastic LSD inputs for deterministic
// parity. Arrays are frame-major; only valid Mask rows are consumed.
type TrainingStepSamples struct {
	Mask               []bool
	Noise              []float32
	DiagonalTime       []float32
	DistillS, DistillT []float32
}

// TrainingStepConfig controls the exact upstream reduction weights. Use
// DefaultTrainingStepConfig for p_equal=0.75 and eos_loss_weight=0.1.
type TrainingStepConfig struct {
	PEqual, EOSLossWeight float32
}

func DefaultTrainingStepConfig() TrainingStepConfig {
	return TrainingStepConfig{PEqual: 0.75, EOSLossWeight: 0.1}
}

type TrainingStepMetrics struct {
	FlowDiagonal, FlowDistill float64
	FlowLoss, EOS, Loss       float64
}

// TrainingStepGradients joins the backbone/conditioning, shared flow-head and
// per-frame learned log-variance leaf gradients.
type TrainingStepGradients struct {
	FlowLM              *FlowLMTrainingGradients
	Flow                *FlowHeadGradients
	Inputs              FlowLMTrainingInputGradients
	DiagonalLogVariance []float32
	DistillLogVariance  []float32
	Weighting           *LSDWeightGradients
}

// PocketTrainingStep evaluates one deterministic one-row upstream training
// step: EOS + normalized LSD diagonal + normalized minimal-stop-gradient s->t.
// Mimi latents, noise, times and log-variance outputs are supplied explicitly.
func PocketTrainingStep(flowLM *FlowLMTrainingCPU, flow *FlowHeadCPU, weighting *LSDWeightMLP, batch FlowLMTrainingBatch, samples TrainingStepSamples, config TrainingStepConfig) (TrainingStepMetrics, TrainingStepGradients, error) {
	if flowLM == nil || flow == nil || weighting == nil || batch.Frames <= 0 || len(samples.Mask) != batch.Frames || !isFinite(config.PEqual) || config.PEqual < 0 || config.PEqual > 1 || !isFinite(config.EOSLossWeight) || config.EOSLossWeight < 0 {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("invalid Pocket TTS training step")
	}
	if err := validateTrainingStepSamples(batch, samples, flowLM.Hidden, flowLM.LatentDim); err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	weightInputElements, ok := checked.MulInt(4, batch.Frames)
	if !ok {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("invalid Pocket TTS weighting shape")
	}
	weightRows, ok := checked.MulInt(2, batch.Frames)
	if !ok {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("invalid Pocket TTS weighting rows")
	}
	zElements, ok := checked.MulInt(batch.Frames, flowLM.Hidden)
	if !ok {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("invalid Pocket TTS training hidden shape")
	}
	weightInputs := make([]float32, weightInputElements)
	for row := 0; row < batch.Frames; row++ {
		weightInputs[2*row], weightInputs[2*row+1] = samples.DiagonalTime[row], samples.DiagonalTime[row]
		base := 2*batch.Frames + 2*row
		weightInputs[base], weightInputs[base+1] = samples.DistillS[row], samples.DistillT[row]
	}
	weightTape, logvars, err := weighting.forward(weightInputs, weightRows)
	if err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	zeroZ := make([]float32, zElements)
	zeroEOS := make([]float32, batch.Frames)
	forward, _, _, err := flowLM.ForwardBackward(batch, zeroZ, zeroEOS)
	if err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	eosLoss, dEOS, err := EOSLossAndGradient(forward.EOS, samples.Mask)
	if err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	for i := range dEOS {
		dEOS[i] *= config.EOSLossWeight
	}
	valid := 0
	for _, active := range samples.Mask {
		if active {
			valid++
		}
	}
	if valid == 0 {
		return TrainingStepMetrics{}, TrainingStepGradients{}, fmt.Errorf("Pocket TTS training step has no valid flow rows")
	}
	invValid := float32(1) / float32(valid)
	dZ := make([]float32, len(forward.Z))
	directTarget := make([]float32, len(batch.NormalizedLatents))
	flowGradients := newFlowHeadGradients(flow)
	dDiagLogvar := make([]float32, batch.Frames)
	dDistillLogvar := make([]float32, batch.Frames)
	metrics := TrainingStepMetrics{EOS: eosLoss}
	diagonalObjective, distillObjective := float64(0), float64(0)
	c, h := flowLM.LatentDim, flowLM.Hidden
	for row, active := range samples.Mask {
		if !active {
			continue
		}
		condition := forward.Z[row*h : (row+1)*h]
		target := batch.NormalizedLatents[row*c : (row+1)*c]
		diagNoise := samples.Noise[row*c : (row+1)*c]
		time := samples.DiagonalTime[row]
		xTime, desired := make([]float32, c), make([]float32, c)
		for i := 0; i < c; i++ {
			xTime[i] = time*target[i] + (1-time)*diagNoise[i]
			desired[i] = target[i] - diagNoise[i]
		}
		tape, prediction, err := flow.forwardTraining(condition, []float32{time, time}, xTime)
		if err != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, err
		}
		diagLoss, dPrediction, dLogvar, err := LSDDiagonalLossAndGradient(prediction, desired, []float32{logvars[row]}, c, true)
		if err != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, err
		}
		diagScale := config.PEqual * invValid
		for i := range dPrediction {
			dPrediction[i] *= diagScale
		}
		rowFlowGradients := newFlowHeadGradients(flow)
		dCondition, _, dXTime, err := flow.backwardTraining(tape, dPrediction, rowFlowGradients)
		if err != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, err
		}
		addFlowHeadGradients(flowGradients, rowFlowGradients, 1)
		for i := range dCondition {
			dZ[row*h+i] += dCondition[i]
		}
		for i := 0; i < c; i++ {
			directTarget[row*c+i] += time*dXTime[i] - dPrediction[i]
		}
		dDiagLogvar[row] = dLogvar[0] * diagScale
		diagonalObjective += diagLoss / float64(valid)
		rawSquare := float64(0)
		for i := range prediction {
			difference := prediction[i] - desired[i]
			rawSquare += float64(difference) * float64(difference)
		}
		metrics.FlowDiagonal += rawSquare / float64(valid)

		distill, distillGradients, err := flow.LSDDistillForwardBackward(condition, samples.DistillS[row], samples.DistillT[row], samples.Noise[row*c:(row+1)*c], target, logvars[batch.Frames+row])
		if err != nil {
			return TrainingStepMetrics{}, TrainingStepGradients{}, err
		}
		distillScale := (1 - config.PEqual) * invValid
		addFlowHeadGradients(flowGradients, distillGradients.Flow, distillScale)
		for i := range distillGradients.Condition {
			dZ[row*h+i] += distillScale * distillGradients.Condition[i]
		}
		for i := 0; i < c; i++ {
			directTarget[row*c+i] += distillScale * distillGradients.Target[i]
		}
		dDistillLogvar[row] = distillScale * distillGradients.DLogVariance
		distillObjective += distill.Loss / float64(valid)
		metrics.FlowDistill += distill.RawSquare / float64(valid)
	}
	dLogvars := append(append([]float32(nil), dDiagLogvar...), dDistillLogvar...)
	weightingGradients, err := weighting.backward(weightTape, dLogvars, weightRows)
	if err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	_, flowLMGradients, inputGradients, err := flowLM.ForwardBackward(batch, dZ, dEOS)
	if err != nil {
		return TrainingStepMetrics{}, TrainingStepGradients{}, err
	}
	addInPlace(inputGradients.NormalizedLatents, directTarget)
	metrics.FlowLoss = float64(config.PEqual)*diagonalObjective + float64(1-config.PEqual)*distillObjective
	metrics.Loss = metrics.FlowLoss + float64(config.EOSLossWeight)*metrics.EOS
	return metrics, TrainingStepGradients{FlowLM: flowLMGradients, Flow: flowGradients, Inputs: inputGradients, DiagonalLogVariance: dDiagLogvar, DistillLogVariance: dDistillLogvar, Weighting: weightingGradients}, nil
}

func validateTrainingStepSamples(batch FlowLMTrainingBatch, samples TrainingStepSamples, hidden, latent int) error {
	_ = hidden
	noiseElements, ok := checked.MulInt(batch.Frames, latent)
	if !ok || len(samples.Noise) != noiseElements || len(samples.DiagonalTime) != batch.Frames || len(samples.DistillS) != batch.Frames || len(samples.DistillT) != batch.Frames {
		return fmt.Errorf("invalid Pocket TTS training sample shape")
	}
	seenPadding := false
	for row, active := range samples.Mask {
		if !active {
			seenPadding = true
		} else if seenPadding {
			return fmt.Errorf("Pocket TTS training mask is not a contiguous prefix")
		}
		for _, value := range []float32{samples.DiagonalTime[row], samples.DistillS[row], samples.DistillT[row]} {
			if !isFinite(value) {
				return fmt.Errorf("Pocket TTS training sample is non-finite")
			}
		}
		if active && (samples.DiagonalTime[row] < 0 || samples.DiagonalTime[row] > 1 || samples.DistillS[row] < 0 || samples.DistillT[row] > 1 || samples.DistillS[row] > samples.DistillT[row]) {
			return fmt.Errorf("Pocket TTS training sample time is invalid")
		}
	}
	for _, values := range [][]float32{samples.Noise} {
		for _, value := range values {
			if !isFinite(value) {
				return fmt.Errorf("Pocket TTS training noise is non-finite")
			}
		}
	}
	return nil
}

func addFlowHeadGradients(dst, src *FlowHeadGradients, scale float32) {
	addLinearGradient(&dst.Input, src.Input, scale)
	addLinearGradient(&dst.Condition, src.Condition, scale)
	for i := range dst.Time {
		addLinearGradient(&dst.Time[i].FC1, src.Time[i].FC1, scale)
		addLinearGradient(&dst.Time[i].FC2, src.Time[i].FC2, scale)
		addScaled(dst.Time[i].RMSWeight, src.Time[i].RMSWeight, scale)
	}
	for i := range dst.Blocks {
		addScaled(dst.Blocks[i].NormWeight, src.Blocks[i].NormWeight, scale)
		addScaled(dst.Blocks[i].NormBias, src.Blocks[i].NormBias, scale)
		addLinearGradient(&dst.Blocks[i].FC1, src.Blocks[i].FC1, scale)
		addLinearGradient(&dst.Blocks[i].FC2, src.Blocks[i].FC2, scale)
		addLinearGradient(&dst.Blocks[i].Modulation, src.Blocks[i].Modulation, scale)
	}
	addLinearGradient(&dst.Final.Linear, src.Final.Linear, scale)
	addLinearGradient(&dst.Final.Modulation, src.Final.Modulation, scale)
}
func addLinearGradient(dst *LinearF32Gradient, src LinearF32Gradient, scale float32) {
	addScaled(dst.Weight, src.Weight, scale)
	addScaled(dst.Bias, src.Bias, scale)
}
func addScaled(dst, src []float32, scale float32) {
	for i, value := range src {
		dst[i] += scale * value
	}
}
