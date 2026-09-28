package pockettts

import (
	"math"
	"testing"
)

func tinyLSDWeighting() *LSDWeightMLP {
	return &LSDWeightMLP{Layers: []LinearF32{tinyLinear(3, 2, .02), tinyLinear(3, 3, -.01), tinyLinear(1, 3, .015)}}
}
func tinyTrainingStepSamples() TrainingStepSamples {
	return TrainingStepSamples{Mask: []bool{true, true, false}, Noise: []float32{.3, -.2, .4, .1, -.5, .2}, DiagonalTime: []float32{.2, .7, .4}, DistillS: []float32{.15, .3, .2}, DistillT: []float32{.8, .65, .9}}
}

func TestPocketTrainingStepFiniteDifference(t *testing.T) {
	flowLM, flow, weighting := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
	flow.Condition = tinyLinear(4, 4, -.02)
	batch, samples, config := tinyFlowLMBatch(), tinyTrainingStepSamples(), DefaultTrainingStepConfig()
	metrics, gradients, err := PocketTrainingStep(flowLM, flow, weighting, batch, samples, config)
	if err != nil {
		t.Fatal(err)
	}
	if !isFinite(float32(metrics.Loss)) || gradients.FlowLM == nil || gradients.Flow == nil || gradients.Weighting == nil {
		t.Fatalf("bad step: %+v", metrics)
	}
	evaluate := func() float64 {
		value, _, e := PocketTrainingStep(flowLM, flow, weighting, batch, samples, config)
		if e != nil {
			t.Fatal(e)
		}
		return value.Loss
	}
	checkCentralDifferenceWithStep(t, "step.audio", batch.NormalizedLatents, gradients.Inputs.NormalizedLatents, evaluate, 2e-4, 4e-2)
	checkCentralDifferenceWithStep(t, "step.voice", batch.VoiceLatents, gradients.Inputs.VoiceLatents, evaluate, 2e-4, 4e-2)
	lmParams, lmGrads := flowLMTrainingParameterPairs(flowLM, gradients.FlowLM)
	for i := range lmParams {
		checkCentralDifferenceWithStep(t, "step."+lmParams[i].name, lmParams[i].values, lmGrads[i].values, evaluate, 2e-4, 5e-2)
	}
	flowParams, flowGrads := flowHeadParameterPairs(flow, gradients.Flow)
	for i := range flowParams {
		checkCentralDifferenceWithStep(t, "step."+flowParams[i].name, flowParams[i].values, flowGrads[i].values, evaluate, 2e-4, 6e-2)
	}
	weightParams, weightGrads := lsdWeightParameterPairs(weighting, gradients.Weighting)
	for i := range weightParams {
		checkCentralDifferenceWithStep(t, "step."+weightParams[i].name, weightParams[i].values, weightGrads[i].values, evaluate, 2e-4, 4e-2)
	}
	if gradients.Inputs.NormalizedLatents[4] != 0 || gradients.Inputs.NormalizedLatents[5] != 0 {
		t.Fatalf("masked final target gradient=%v", gradients.Inputs.NormalizedLatents)
	}
}

func TestPocketTrainingStepRejectsMalformed(t *testing.T) {
	flowLM, flow, weighting := tinyFlowLMTraining(), tinyTrainableFlowHead(), tinyLSDWeighting()
	flow.Condition = tinyLinear(4, 4, -.02)
	batch, samples := tinyFlowLMBatch(), tinyTrainingStepSamples()
	samples.Mask = []bool{true, false, true}
	if _, _, err := PocketTrainingStep(flowLM, flow, weighting, batch, samples, DefaultTrainingStepConfig()); err == nil {
		t.Fatal("accepted non-prefix mask")
	}
	samples = tinyTrainingStepSamples()
	overflowFrames := int(^uint(0)>>1)/2 + 1
	batch.Frames = overflowFrames
	if _, _, err := PocketTrainingStep(flowLM, flow, weighting, batch, samples, DefaultTrainingStepConfig()); err == nil {
		t.Fatal("accepted overflowing training step shape")
	}
	batch = tinyFlowLMBatch()
	samples = tinyTrainingStepSamples()
	samples.DistillT[0] = float32(math.NaN())
	if _, _, err := PocketTrainingStep(flowLM, flow, weighting, batch, samples, DefaultTrainingStepConfig()); err == nil {
		t.Fatal("accepted non-finite time")
	}
}

func lsdWeightParameterPairs(model *LSDWeightMLP, gradients *LSDWeightGradients) ([]parameterPair, []parameterPair) {
	params, grads := []parameterPair{}, []parameterPair{}
	for i := range model.Layers {
		params = append(params, parameterPair{"weighting.weight", model.Layers[i].Weight}, parameterPair{"weighting.bias", model.Layers[i].Bias})
		grads = append(grads, parameterPair{"weighting.weight", gradients.Layers[i].Weight}, parameterPair{"weighting.bias", gradients.Layers[i].Bias})
	}
	return params, grads
}
