package pockettts

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestFlowHeadTimeJVPBackwardFiniteDifference(t *testing.T) {
	model := tinyTrainableFlowHead()
	condition := []float32{.2, -.3, .5}
	times := []float32{.25, .8}
	input := []float32{-.4, .7}
	dOutput := []float32{.6, -.9}
	dTangent := []float32{-.35, .45}
	_, _, gradients, dCondition, dTimes, dInput, err := model.ForwardTimeJVPBackward(condition, times, input, 1, dOutput, dTangent)
	if err != nil {
		t.Fatal(err)
	}
	evaluate := func() float64 {
		output, tangent, e := model.ForwardTimeJVP(condition, times, input, 1)
		if e != nil {
			t.Fatal(e)
		}
		value := float64(0)
		for i := range output {
			value += float64(output[i])*float64(dOutput[i]) + float64(tangent[i])*float64(dTangent[i])
		}
		return value
	}
	checkCentralDifference(t, "mixed.input", input, dInput, evaluate, 6e-3)
	checkCentralDifference(t, "mixed.condition", condition, dCondition, evaluate, 8e-3)
	checkCentralDifference(t, "mixed.times", times, dTimes, evaluate, 8e-3)
	params, grads := flowHeadParameterPairs(model, gradients)
	for i := range params {
		checkCentralDifference(t, "mixed."+params[i].name, params[i].values, grads[i].values, evaluate, 1.2e-2)
	}
}

func TestFlowHeadTimeJVPBackwardOrdinarySeedParity(t *testing.T) {
	model := tinyTrainableFlowHead()
	condition := []float32{.2, -.3, .5}
	times := []float32{.25, .8}
	input := []float32{-.4, .7}
	dOutput := []float32{.6, -.9}
	output, _, mixed, dCondition, dTimes, dInput, err := model.ForwardTimeJVPBackward(condition, times, input, 1, dOutput, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	wantOutput, ordinary, wantCondition, wantTimes, wantInput, err := model.ForwardBackward(condition, times, input, dOutput)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "mixed ordinary output", output, wantOutput, 1e-6)
	assertSliceClose(t, "mixed ordinary condition", dCondition, wantCondition, 2e-6)
	assertSliceClose(t, "mixed ordinary times", dTimes, wantTimes, 2e-6)
	assertSliceClose(t, "mixed ordinary input", dInput, wantInput, 2e-6)
	for name, got := range flowHeadGradientMap(mixed) {
		assertSliceClose(t, "mixed ordinary "+name, got, flowHeadGradientMap(ordinary)[name], 2e-6)
	}
}

func TestFlowHeadTimeJVPBackwardPyTorchParity(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowHeadBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit || len(oracle.Mixed) != len(oracle.Times) || len(oracle.JVPTimes) != len(oracle.Times) {
		t.Fatalf("unexpected mixed oracle schema/revision/shape")
	}
	model := flowHeadFromOracle(t, oracle.Parameters)
	for _, mixed := range oracle.Mixed {
		output, tangent, gradients, dCondition, dTimes, dInput, err := model.ForwardTimeJVPBackward(oracle.Condition, oracle.Times, oracle.Input, mixed.TimeIndex, mixed.DOutput, mixed.DTangent)
		if err != nil {
			t.Fatal(err)
		}
		assertSliceClose(t, "mixed PyTorch output", output, oracle.Output, 3e-6)
		assertSliceClose(t, "mixed PyTorch tangent", tangent, oracle.JVPTimes[mixed.TimeIndex], 5e-6)
		assertSliceClose(t, "mixed PyTorch condition", dCondition, mixed.DCondition, 8e-6)
		assertSliceClose(t, "mixed PyTorch times", dTimes, mixed.DTimes, 8e-6)
		assertSliceClose(t, "mixed PyTorch input", dInput, mixed.DInput, 8e-6)
		gradientMap := flowHeadGradientMap(gradients)
		assertFlowHeadGradientCoverage(t, oracle, gradientMap)
		for name, got := range gradientMap {
			assertSliceClose(t, "mixed PyTorch "+name, got, mixed.Gradients[name], 1e-5)
		}
	}
}

func TestFlowHeadTimeJVPBackwardRejectsMalformed(t *testing.T) {
	model := tinyTrainableFlowHead()
	if _, _, _, _, _, _, err := model.ForwardTimeJVPBackward([]float32{0, 0, 0}, []float32{0, 0}, []float32{0, 0}, 2, []float32{0, 0}, []float32{0, 0}); err == nil {
		t.Fatal("accepted mixed derivative index")
	}
	if _, _, _, _, _, _, err := model.ForwardTimeJVPBackward([]float32{0, 0, 0}, []float32{0, 0}, []float32{0, 0}, 0, []float32{0}, []float32{0, 0}); err == nil {
		t.Fatal("accepted mixed derivative seed shape")
	}
	if _, _, _, _, _, _, err := model.ForwardTimeJVPBackward([]float32{0, 0, 0}, []float32{0, 0}, []float32{0, 0}, 0, []float32{0, 0}, []float32{float32(math.NaN()), 0}); err == nil {
		t.Fatal("accepted non-finite mixed derivative seed")
	}
}
