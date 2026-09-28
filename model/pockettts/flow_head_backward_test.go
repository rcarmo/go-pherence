package pockettts

import (
	"math"
	"testing"
)

func tinyLinear(out, in int, start float32) LinearF32 {
	weight := make([]float32, out*in)
	bias := make([]float32, out)
	for i := range weight {
		weight[i] = start + float32((i*7)%13-6)*0.025
	}
	for i := range bias {
		bias[i] = start/3 + float32(i-1)*0.02
	}
	return LinearF32{Weight: weight, Bias: bias, In: in, Out: out}
}

func tinyTrainableFlowHead() *FlowHeadCPU {
	const d = 4
	return &FlowHeadCPU{
		Input:     tinyLinear(d, 2, .03),
		Condition: tinyLinear(d, 3, -.02),
		Time: []TimestepMLP{
			{Frequencies: []float32{1, .1}, FC1: tinyLinear(d, d, .01), FC2: tinyLinear(d, d, -.03), RMSWeight: []float32{.8, 1.1, .9, 1.2}, RMSEpsilon: 1e-5},
			{Frequencies: []float32{1, .1}, FC1: tinyLinear(d, d, -.015), FC2: tinyLinear(d, d, .025), RMSWeight: []float32{1.05, .95, 1.15, .85}, RMSEpsilon: 1e-5},
		},
		Blocks: []AdaLNResidual{
			{NormWeight: []float32{1.1, .9, 1.2, .8}, NormBias: []float32{.02, -.03, .01, .04}, FC1: tinyLinear(d, d, .02), FC2: tinyLinear(d, d, -.01), Modulation: tinyLinear(3*d, d, .005), Epsilon: 1e-6},
			{NormWeight: []float32{.95, 1.05, .85, 1.15}, NormBias: []float32{-.01, .02, -.04, .03}, FC1: tinyLinear(d, d, -.025), FC2: tinyLinear(d, d, .015), Modulation: tinyLinear(3*d, d, -.004), Epsilon: 1e-6},
		},
		Final: AdaLNFinal{Linear: tinyLinear(2, d, .02), Modulation: tinyLinear(2*d, d, -.006), Epsilon: 1e-6},
	}
}

func TestFlowHeadBackwardFiniteDifference(t *testing.T) {
	model := tinyTrainableFlowHead()
	condition := []float32{.2, -.3, .5}
	times := []float32{.25, .8}
	input := []float32{-.4, .7}
	upstream := []float32{.6, -.9}
	output, gradients, dCondition, dTimes, dInput, err := model.ForwardBackward(condition, times, input, upstream)
	if err != nil {
		t.Fatal(err)
	}
	if err := model.Forward(nil, nil, nil, nil); err == nil {
		t.Fatal("malformed inference call unexpectedly passed")
	}
	forward := make([]float32, 2)
	if err = model.Forward(forward, condition, times, input); err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "forward", output, forward, 2e-5)

	evaluate := func() float64 {
		result := make([]float32, 2)
		if e := model.Forward(result, condition, times, input); e != nil {
			t.Fatal(e)
		}
		return float64(result[0])*float64(upstream[0]) + float64(result[1])*float64(upstream[1])
	}
	checkCentralDifference(t, "input", input, dInput, evaluate, 2e-3)
	checkCentralDifference(t, "condition", condition, dCondition, evaluate, 3e-3)
	checkCentralDifference(t, "times", times, dTimes, evaluate, 3e-3)

	params, grads := flowHeadParameterPairs(model, gradients)
	for i := range params {
		checkCentralDifference(t, params[i].name, params[i].values, grads[i].values, evaluate, 4e-3)
	}
}

func TestFlowHeadBackwardRejectsBF16AndNonfinite(t *testing.T) {
	model := tinyTrainableFlowHead()
	model.Input.WeightBF16 = make([]uint16, len(model.Input.Weight))
	if _, _, _, _, _, err := model.ForwardBackward([]float32{0, 0, 0}, []float32{0, 0}, []float32{0, 0}, []float32{0, 0}); err == nil {
		t.Fatal("accepted BF16 training weight")
	}
	model = tinyTrainableFlowHead()
	if _, _, _, _, _, err := model.ForwardBackward([]float32{0, 0, 0}, []float32{0, float32(math.NaN())}, []float32{0, 0}, []float32{0, 0}); err == nil {
		t.Fatal("accepted non-finite time")
	}
	model = tinyTrainableFlowHead()
	model.Blocks[0].Modulation.Out--
	if _, _, _, _, _, err := model.ForwardBackward([]float32{0, 0, 0}, []float32{0, 0}, []float32{0, 0}, []float32{0, 0}); err == nil {
		t.Fatal("accepted malformed topology")
	}
}

type parameterPair struct {
	name   string
	values []float32
}

func flowHeadParameterPairs(model *FlowHeadCPU, gradients *FlowHeadGradients) ([]parameterPair, []parameterPair) {
	params := []parameterPair{{"input.weight", model.Input.Weight}, {"input.bias", model.Input.Bias}, {"condition.weight", model.Condition.Weight}, {"condition.bias", model.Condition.Bias}}
	grads := []parameterPair{{"input.weight", gradients.Input.Weight}, {"input.bias", gradients.Input.Bias}, {"condition.weight", gradients.Condition.Weight}, {"condition.bias", gradients.Condition.Bias}}
	for i := range model.Time {
		params = append(params,
			parameterPair{"time.fc1.weight", model.Time[i].FC1.Weight}, parameterPair{"time.fc1.bias", model.Time[i].FC1.Bias},
			parameterPair{"time.fc2.weight", model.Time[i].FC2.Weight}, parameterPair{"time.fc2.bias", model.Time[i].FC2.Bias},
			parameterPair{"time.rms", model.Time[i].RMSWeight})
		grads = append(grads,
			parameterPair{"time.fc1.weight", gradients.Time[i].FC1.Weight}, parameterPair{"time.fc1.bias", gradients.Time[i].FC1.Bias},
			parameterPair{"time.fc2.weight", gradients.Time[i].FC2.Weight}, parameterPair{"time.fc2.bias", gradients.Time[i].FC2.Bias},
			parameterPair{"time.rms", gradients.Time[i].RMSWeight})
	}
	for i := range model.Blocks {
		params = append(params,
			parameterPair{"block.norm.weight", model.Blocks[i].NormWeight}, parameterPair{"block.norm.bias", model.Blocks[i].NormBias},
			parameterPair{"block.fc1.weight", model.Blocks[i].FC1.Weight}, parameterPair{"block.fc1.bias", model.Blocks[i].FC1.Bias},
			parameterPair{"block.fc2.weight", model.Blocks[i].FC2.Weight}, parameterPair{"block.fc2.bias", model.Blocks[i].FC2.Bias},
			parameterPair{"block.mod.weight", model.Blocks[i].Modulation.Weight}, parameterPair{"block.mod.bias", model.Blocks[i].Modulation.Bias})
		grads = append(grads,
			parameterPair{"block.norm.weight", gradients.Blocks[i].NormWeight}, parameterPair{"block.norm.bias", gradients.Blocks[i].NormBias},
			parameterPair{"block.fc1.weight", gradients.Blocks[i].FC1.Weight}, parameterPair{"block.fc1.bias", gradients.Blocks[i].FC1.Bias},
			parameterPair{"block.fc2.weight", gradients.Blocks[i].FC2.Weight}, parameterPair{"block.fc2.bias", gradients.Blocks[i].FC2.Bias},
			parameterPair{"block.mod.weight", gradients.Blocks[i].Modulation.Weight}, parameterPair{"block.mod.bias", gradients.Blocks[i].Modulation.Bias})
	}
	params = append(params,
		parameterPair{"final.linear.weight", model.Final.Linear.Weight}, parameterPair{"final.linear.bias", model.Final.Linear.Bias},
		parameterPair{"final.mod.weight", model.Final.Modulation.Weight}, parameterPair{"final.mod.bias", model.Final.Modulation.Bias})
	grads = append(grads,
		parameterPair{"final.linear.weight", gradients.Final.Linear.Weight}, parameterPair{"final.linear.bias", gradients.Final.Linear.Bias},
		parameterPair{"final.mod.weight", gradients.Final.Modulation.Weight}, parameterPair{"final.mod.bias", gradients.Final.Modulation.Bias})
	return params, grads
}

func checkCentralDifference(t *testing.T, name string, values, gradients []float32, evaluate func() float64, tolerance float64) {
	t.Helper()
	if len(values) != len(gradients) {
		t.Fatalf("%s shape=%d gradient=%d", name, len(values), len(gradients))
	}
	const step = float32(2e-3)
	for i := range values {
		original := values[i]
		values[i] = original + step
		plus := evaluate()
		values[i] = original - step
		minus := evaluate()
		values[i] = original
		numeric := (plus - minus) / (2 * float64(step))
		if math.Abs(float64(gradients[i])-numeric) > tolerance {
			t.Fatalf("%s[%d] gradient=%g numeric=%g diff=%g", name, i, gradients[i], numeric, math.Abs(float64(gradients[i])-numeric))
		}
	}
}
