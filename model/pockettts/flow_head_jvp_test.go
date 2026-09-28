package pockettts

import (
	"encoding/json"
	"os"
	"testing"
)

func TestFlowHeadTimeJVPCentralDifference(t *testing.T) {
	model := tinyTrainableFlowHead()
	condition := []float32{.2, -.3, .5}
	times := []float32{.25, .8}
	input := []float32{-.4, .7}
	for timeIndex := range times {
		output, tangent, err := model.ForwardTimeJVP(condition, times, input, timeIndex)
		if err != nil {
			t.Fatal(err)
		}
		forward := make([]float32, 2)
		if err = model.Forward(forward, condition, times, input); err != nil {
			t.Fatal(err)
		}
		assertSliceClose(t, "JVP output", output, forward, 2e-5)
		const step = float32(1e-3)
		original := times[timeIndex]
		times[timeIndex] = original + step
		plus := make([]float32, 2)
		if err = model.Forward(plus, condition, times, input); err != nil {
			t.Fatal(err)
		}
		times[timeIndex] = original - step
		minus := make([]float32, 2)
		if err = model.Forward(minus, condition, times, input); err != nil {
			t.Fatal(err)
		}
		times[timeIndex] = original
		for i := range tangent {
			numeric := (plus[i] - minus[i]) / (2 * step)
			if diff := absF32(tangent[i] - numeric); diff > 2e-3 {
				t.Fatalf("time %d output %d JVP=%g numeric=%g diff=%g", timeIndex, i, tangent[i], numeric, diff)
			}
		}
	}
}

func TestFlowHeadTimeJVPPyTorchParity(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowHeadBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	model := flowHeadFromOracle(t, oracle.Parameters)
	if len(oracle.JVPTimes) != len(oracle.Times) {
		t.Fatalf("oracle JVP rows=%d want=%d", len(oracle.JVPTimes), len(oracle.Times))
	}
	for timeIndex := range oracle.Times {
		_, tangent, err := model.ForwardTimeJVP(oracle.Condition, oracle.Times, oracle.Input, timeIndex)
		if err != nil {
			t.Fatal(err)
		}
		assertSliceClose(t, "PyTorch time JVP", tangent, oracle.JVPTimes[timeIndex], 5e-6)
		// Reverse-mode dTimes also checks the JVP/VJP contraction.
		dot := float32(0)
		for i := range tangent {
			dot += tangent[i] * oracle.DOutput[i]
		}
		if diff := absF32(dot - oracle.DTimes[timeIndex]); diff > 5e-6 {
			t.Fatalf("time %d JVP VJP=%g PyTorch=%g diff=%g tangent=%v", timeIndex, dot, oracle.DTimes[timeIndex], diff, tangent)
		}
	}
}

func TestFlowHeadTimeJVPRejectsMalformed(t *testing.T) {
	if _, _, err := (*FlowHeadCPU)(nil).ForwardTimeJVP(nil, nil, nil, 0); err == nil {
		t.Fatal("accepted nil time JVP head")
	}
	model := tinyTrainableFlowHead()
	if _, _, err := model.ForwardTimeJVP([]float32{0, 0, 0}, []float32{0, 0}, []float32{0, 0}, 2); err == nil {
		t.Fatal("accepted out-of-range time JVP index")
	}
	model = tinyTrainableFlowHead()
	model.Input.WeightBF16 = make([]uint16, len(model.Input.Weight))
	if _, _, err := model.ForwardTimeJVP([]float32{0, 0, 0}, []float32{0, 0}, []float32{0, 0}, 0); err == nil {
		t.Fatal("accepted BF16 time JVP head")
	}
	model = tinyTrainableFlowHead()
	model.Final.Linear.Out = -1
	if _, _, err := model.ForwardTimeJVP([]float32{0, 0, 0}, []float32{0, 0}, []float32{0, 0}, 0); err == nil {
		t.Fatal("accepted negative time JVP output shape")
	}
}

func absF32(value float32) float32 {
	if value < 0 {
		return -value
	}
	return value
}
