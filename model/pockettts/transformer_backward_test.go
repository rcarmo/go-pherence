package pockettts

import (
	"math"
	"testing"
)

func tinyTrainableTransformer() *TransformerCPU {
	const width = 4
	biasFree := func(out, in int, start float32) LinearF32 {
		linear := tinyLinear(out, in, start)
		linear.Bias = nil
		return linear
	}
	return &TransformerCPU{
		Width: 4, Heads: 2, HeadDim: 2, Context: 2, MaxPeriod: 100,
		Layers: []TransformerLayerCPU{{
			Norm1Weight: []float32{1.1, .9, 1.2, .8}, Norm1Bias: []float32{.02, -.03, .01, .04},
			Norm2Weight: []float32{.95, 1.05, .85, 1.15}, Norm2Bias: []float32{-.01, .02, -.04, .03},
			InProjection: biasFree(3*width, width, .01), OutProjection: biasFree(width, width, -.02),
			FC1: biasFree(6, width, .03), FC2: biasFree(width, 6, -.015),
			LayerScale1: []float32{.8, 1.1, .9, 1.2}, LayerScale2: []float32{1.05, .95, 1.15, .85},
		}},
		FinalWeight: []float32{1.02, .98, 1.08, .92}, FinalBias: []float32{.01, -.02, .03, -.04},
	}
}

func TestTransformerBackwardFiniteDifference(t *testing.T) {
	model := tinyTrainableTransformer()
	sequence := []float32{.2, -.4, .1, .5, -.3, .7, .6, -.2, .8, -.1, .4, -.5}
	dOutput := []float32{.1, -.2, .3, -.4, .5, -.6, .7, -.8, -.3, .4, -.5, .6}
	output, gradients, dSequence, err := model.ForwardBackward(sequence, dOutput, 3)
	if err != nil {
		t.Fatal(err)
	}
	forward, err := model.Forward(sequence, 3)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "transformer forward", output, forward, 3e-5)
	evaluate := func() float64 {
		value, e := model.Forward(sequence, 3)
		if e != nil {
			t.Fatal(e)
		}
		total := float64(0)
		for i := range value {
			total += float64(value[i]) * float64(dOutput[i])
		}
		return total
	}
	checkCentralDifference(t, "transformer.input", sequence, dSequence, evaluate, 1.5e-2)
	params, grads := transformerParameterPairs(model, gradients)
	for i := range params {
		checkCentralDifference(t, params[i].name, params[i].values, grads[i].values, evaluate, 2e-2)
	}
}

func TestTransformerBackwardWithoutLayerScaleOrFinalNorm(t *testing.T) {
	model := tinyTrainableTransformer()
	model.Context = 0
	model.Layers[0].LayerScale1, model.Layers[0].LayerScale2 = nil, nil
	model.FinalWeight, model.FinalBias = nil, nil
	sequence := []float32{.2, -.4, .1, .5, -.3, .7, .6, -.2}
	dOutput := []float32{.1, -.2, .3, -.4, .5, -.6, .7, -.8}
	output, _, _, err := model.ForwardBackward(sequence, dOutput, 2)
	if err != nil {
		t.Fatal(err)
	}
	forward, err := model.Forward(sequence, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "transformer optional forward", output, forward, 3e-5)
}

func TestTransformerBackwardRejectsMalformed(t *testing.T) {
	model := tinyTrainableTransformer()
	model.Layers[0].InProjection.WeightBF16 = make([]uint16, len(model.Layers[0].InProjection.Weight))
	if _, _, _, err := model.ForwardBackward(make([]float32, 12), make([]float32, 12), 3); err == nil {
		t.Fatal("accepted BF16 transformer training")
	}
	model = tinyTrainableTransformer()
	model.Heads = 3
	if _, _, _, err := model.ForwardBackward(make([]float32, 12), make([]float32, 12), 3); err == nil {
		t.Fatal("accepted malformed heads")
	}
	model = tinyTrainableTransformer()
	overflowRows := int(^uint(0)>>1)/model.Width + 1
	if _, _, _, err := model.ForwardBackward(nil, nil, overflowRows); err == nil {
		t.Fatal("accepted overflowing transformer shape")
	}
	if _, err := model.Forward(nil, overflowRows); err == nil {
		t.Fatal("stateless forward accepted overflowing transformer shape")
	}
	model = tinyTrainableTransformer()
	input := make([]float32, 12)
	input[2] = float32(math.NaN())
	if _, _, _, err := model.ForwardBackward(input, make([]float32, 12), 3); err == nil {
		t.Fatal("accepted non-finite transformer input")
	}
}

func transformerParameterPairs(model *TransformerCPU, gradients *TransformerGradients) ([]parameterPair, []parameterPair) {
	params, grads := []parameterPair{}, []parameterPair{}
	for i := range model.Layers {
		layer, gradient := &model.Layers[i], &gradients.Layers[i]
		params = append(params,
			parameterPair{"transformer.norm1.weight", layer.Norm1Weight}, parameterPair{"transformer.norm1.bias", layer.Norm1Bias},
			parameterPair{"transformer.norm2.weight", layer.Norm2Weight}, parameterPair{"transformer.norm2.bias", layer.Norm2Bias},
			parameterPair{"transformer.in.weight", layer.InProjection.Weight}, parameterPair{"transformer.out.weight", layer.OutProjection.Weight},
			parameterPair{"transformer.fc1.weight", layer.FC1.Weight}, parameterPair{"transformer.fc2.weight", layer.FC2.Weight})
		grads = append(grads,
			parameterPair{"transformer.norm1.weight", gradient.Norm1Weight}, parameterPair{"transformer.norm1.bias", gradient.Norm1Bias},
			parameterPair{"transformer.norm2.weight", gradient.Norm2Weight}, parameterPair{"transformer.norm2.bias", gradient.Norm2Bias},
			parameterPair{"transformer.in.weight", gradient.InProjection.Weight}, parameterPair{"transformer.out.weight", gradient.OutProjection.Weight},
			parameterPair{"transformer.fc1.weight", gradient.FC1.Weight}, parameterPair{"transformer.fc2.weight", gradient.FC2.Weight})
		if layer.LayerScale1 != nil {
			params = append(params, parameterPair{"transformer.scale1", layer.LayerScale1})
			grads = append(grads, parameterPair{"transformer.scale1", gradient.LayerScale1})
		}
		if layer.LayerScale2 != nil {
			params = append(params, parameterPair{"transformer.scale2", layer.LayerScale2})
			grads = append(grads, parameterPair{"transformer.scale2", gradient.LayerScale2})
		}
	}
	if model.FinalWeight != nil {
		params = append(params, parameterPair{"transformer.final.weight", model.FinalWeight}, parameterPair{"transformer.final.bias", model.FinalBias})
		grads = append(grads, parameterPair{"transformer.final.weight", gradients.FinalWeight}, parameterPair{"transformer.final.bias", gradients.FinalBias})
	}
	return params, grads
}
