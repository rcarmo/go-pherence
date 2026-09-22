package pockettts

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"
)

type flowHeadBackwardOracle struct {
	Schema           int                  `json:"schema"`
	UpstreamRevision string               `json:"upstream_revision"`
	Condition        []float32            `json:"condition"`
	Times            []float32            `json:"times"`
	Input            []float32            `json:"input"`
	DOutput          []float32            `json:"d_output"`
	Output           []float32            `json:"output"`
	DCondition       []float32            `json:"d_condition"`
	DTimes           []float32            `json:"d_times"`
	JVPTimes         [][]float32          `json:"jvp_times"`
	DInput           []float32            `json:"d_input"`
	Parameters       map[string][]float32 `json:"parameters"`
	Gradients        map[string][]float32 `json:"gradients"`
	Mixed            []struct {
		TimeIndex  int                  `json:"time_index"`
		DOutput    []float32            `json:"d_output"`
		DTangent   []float32            `json:"d_tangent"`
		Loss       float32              `json:"loss"`
		DCondition []float32            `json:"d_condition"`
		DTimes     []float32            `json:"d_times"`
		DInput     []float32            `json:"d_input"`
		Gradients  map[string][]float32 `json:"gradients"`
	} `json:"mixed"`
}

func TestFlowHeadBackwardPyTorchParity(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowHeadBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit {
		t.Fatalf("unexpected oracle schema/revision: %+v", oracle)
	}
	model := flowHeadFromOracle(t, oracle.Parameters)
	output, gradients, dCondition, dTimes, dInput, err := model.ForwardBackward(oracle.Condition, oracle.Times, oracle.Input, oracle.DOutput)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "output", output, oracle.Output, 3e-6)
	assertSliceClose(t, "d_condition", dCondition, oracle.DCondition, 4e-6)
	assertSliceClose(t, "d_times", dTimes, oracle.DTimes, 4e-6)
	assertSliceClose(t, "d_input", dInput, oracle.DInput, 4e-6)
	gradientMap := flowHeadGradientMap(gradients)
	assertFlowHeadGradientCoverage(t, oracle, gradientMap)
	for name, got := range gradientMap {
		want, ok := oracle.Gradients[name]
		if !ok {
			t.Fatalf("oracle lacks gradient %q", name)
		}
		assertSliceClose(t, name, got, want, 5e-6)
	}
}

func flowHeadFromOracle(t *testing.T, p map[string][]float32) *FlowHeadCPU {
	t.Helper()
	linear := func(prefix string, out, in int) LinearF32 {
		weight, ok := p[prefix+".weight"]
		if !ok || len(weight) != out*in {
			t.Fatalf("bad oracle linear %s weight=%d", prefix, len(weight))
		}
		bias, ok := p[prefix+".bias"]
		if !ok || len(bias) != out {
			t.Fatalf("bad oracle linear %s bias=%d", prefix, len(bias))
		}
		return LinearF32{Weight: append([]float32(nil), weight...), Bias: append([]float32(nil), bias...), In: in, Out: out}
	}
	model := &FlowHeadCPU{
		Input:     linear("input_proj", 4, 2),
		Condition: linear("cond_embed", 4, 3),
		Time:      make([]TimestepMLP, 2),
		Blocks:    make([]AdaLNResidual, 2),
		Final: AdaLNFinal{
			Linear:     linear("final_layer.linear", 2, 4),
			Modulation: linear("final_layer.adaLN_modulation.1", 8, 4),
			Epsilon:    1e-6,
		},
	}
	for i := range model.Time {
		prefix := fmt.Sprintf("time_embed.%d", i)
		model.Time[i] = TimestepMLP{
			Frequencies: append([]float32(nil), p[prefix+".freqs"]...),
			FC1:         linear(prefix+".mlp.0", 4, 4),
			FC2:         linear(prefix+".mlp.2", 4, 4),
			RMSWeight:   append([]float32(nil), p[prefix+".mlp.3.alpha"]...),
			RMSEpsilon:  1e-5,
		}
	}
	for i := range model.Blocks {
		prefix := fmt.Sprintf("res_blocks.%d", i)
		model.Blocks[i] = AdaLNResidual{
			NormWeight: append([]float32(nil), p[prefix+".in_ln.weight"]...),
			NormBias:   append([]float32(nil), p[prefix+".in_ln.bias"]...),
			FC1:        linear(prefix+".mlp.0", 4, 4),
			FC2:        linear(prefix+".mlp.2", 4, 4),
			Modulation: linear(prefix+".adaLN_modulation.1", 12, 4),
			Epsilon:    1e-6,
		}
	}
	return model
}

func flowHeadGradientMap(g *FlowHeadGradients) map[string][]float32 {
	out := map[string][]float32{
		"input_proj.weight":                     g.Input.Weight,
		"input_proj.bias":                       g.Input.Bias,
		"cond_embed.weight":                     g.Condition.Weight,
		"cond_embed.bias":                       g.Condition.Bias,
		"final_layer.linear.weight":             g.Final.Linear.Weight,
		"final_layer.linear.bias":               g.Final.Linear.Bias,
		"final_layer.adaLN_modulation.1.weight": g.Final.Modulation.Weight,
		"final_layer.adaLN_modulation.1.bias":   g.Final.Modulation.Bias,
	}
	for i := range g.Time {
		prefix := fmt.Sprintf("time_embed.%d", i)
		out[prefix+".mlp.0.weight"] = g.Time[i].FC1.Weight
		out[prefix+".mlp.0.bias"] = g.Time[i].FC1.Bias
		out[prefix+".mlp.2.weight"] = g.Time[i].FC2.Weight
		out[prefix+".mlp.2.bias"] = g.Time[i].FC2.Bias
		out[prefix+".mlp.3.alpha"] = g.Time[i].RMSWeight
	}
	for i := range g.Blocks {
		prefix := fmt.Sprintf("res_blocks.%d", i)
		out[prefix+".in_ln.weight"] = g.Blocks[i].NormWeight
		out[prefix+".in_ln.bias"] = g.Blocks[i].NormBias
		out[prefix+".mlp.0.weight"] = g.Blocks[i].FC1.Weight
		out[prefix+".mlp.0.bias"] = g.Blocks[i].FC1.Bias
		out[prefix+".mlp.2.weight"] = g.Blocks[i].FC2.Weight
		out[prefix+".mlp.2.bias"] = g.Blocks[i].FC2.Bias
		out[prefix+".adaLN_modulation.1.weight"] = g.Blocks[i].Modulation.Weight
		out[prefix+".adaLN_modulation.1.bias"] = g.Blocks[i].Modulation.Bias
	}
	return out
}

func assertFlowHeadGradientCoverage(t *testing.T, oracle flowHeadBackwardOracle, got map[string][]float32) {
	t.Helper()
	for name := range oracle.Gradients {
		if _, ok := got[name]; !ok {
			t.Fatalf("missing native gradient %q", name)
		}
	}
	for name := range got {
		if _, ok := oracle.Gradients[name]; !ok && !strings.HasSuffix(name, ".freqs") {
			t.Fatalf("unexpected native gradient %q", name)
		}
	}
}
