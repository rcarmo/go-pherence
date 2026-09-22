package pockettts

import (
	"encoding/json"
	"fmt"
	"os"
	"testing"
)

type transformerBackwardOracle struct {
	Schema           int                  `json:"schema"`
	UpstreamRevision string               `json:"upstream_revision"`
	Rows             int                  `json:"rows"`
	Width            int                  `json:"width"`
	Heads            int                  `json:"heads"`
	Context          int                  `json:"context"`
	MaxPeriod        float64              `json:"max_period"`
	Sequence         []float32            `json:"sequence"`
	DOutput          []float32            `json:"d_output"`
	Output           []float32            `json:"output"`
	DSequence        []float32            `json:"d_sequence"`
	Parameters       map[string][]float32 `json:"parameters"`
	Gradients        map[string][]float32 `json:"gradients"`
}

func TestTransformerBackwardPyTorchParity(t *testing.T) {
	data, err := os.ReadFile("testdata/transformer_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle transformerBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit || oracle.Rows != 3 || oracle.Width != 4 {
		t.Fatal("unexpected transformer oracle")
	}
	model := transformerFromOracle(t, oracle)
	output, gradients, dSequence, err := model.ForwardBackward(oracle.Sequence, oracle.DOutput, oracle.Rows)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "transformer PyTorch output", output, oracle.Output, 4e-6)
	assertSliceClose(t, "transformer PyTorch input", dSequence, oracle.DSequence, 8e-6)
	gradientMap := transformerGradientMap(gradients)
	if len(gradientMap) != len(oracle.Gradients) {
		t.Fatalf("transformer gradients=%d want=%d", len(gradientMap), len(oracle.Gradients))
	}
	for name, got := range gradientMap {
		want, ok := oracle.Gradients[name]
		if !ok {
			t.Fatalf("oracle lacks %s", name)
		}
		assertSliceClose(t, "transformer PyTorch "+name, got, want, 1e-5)
	}
}

func transformerFromOracle(t testing.TB, oracle transformerBackwardOracle) *TransformerCPU {
	t.Helper()
	p := oracle.Parameters
	linear := func(prefix string, out, in int) LinearF32 {
		values := p[prefix+".weight"]
		if len(values) != out*in {
			t.Fatalf("%s values=%d", prefix, len(values))
		}
		return LinearF32{Weight: append([]float32(nil), values...), In: in, Out: out}
	}
	prefix := "transformer.layers.0"
	return &TransformerCPU{
		Width: oracle.Width, Heads: oracle.Heads, HeadDim: oracle.Width / oracle.Heads, Context: oracle.Context, MaxPeriod: oracle.MaxPeriod,
		Layers: []TransformerLayerCPU{{
			Norm1Weight: append([]float32(nil), p[prefix+".norm1.weight"]...), Norm1Bias: append([]float32(nil), p[prefix+".norm1.bias"]...),
			Norm2Weight: append([]float32(nil), p[prefix+".norm2.weight"]...), Norm2Bias: append([]float32(nil), p[prefix+".norm2.bias"]...),
			InProjection: linear(prefix+".self_attn.in_proj", 3*oracle.Width, oracle.Width), OutProjection: linear(prefix+".self_attn.out_proj", oracle.Width, oracle.Width),
			FC1: linear(prefix+".linear1", 6, oracle.Width), FC2: linear(prefix+".linear2", oracle.Width, 6),
			LayerScale1: append([]float32(nil), p[prefix+".layer_scale_1.scale"]...), LayerScale2: append([]float32(nil), p[prefix+".layer_scale_2.scale"]...),
		}},
		FinalWeight: append([]float32(nil), p["final.weight"]...), FinalBias: append([]float32(nil), p["final.bias"]...),
	}
}

func transformerGradientMap(g *TransformerGradients) map[string][]float32 {
	out := map[string][]float32{"final.weight": g.FinalWeight, "final.bias": g.FinalBias}
	for i, layer := range g.Layers {
		prefix := fmt.Sprintf("transformer.layers.%d", i)
		out[prefix+".norm1.weight"], out[prefix+".norm1.bias"] = layer.Norm1Weight, layer.Norm1Bias
		out[prefix+".norm2.weight"], out[prefix+".norm2.bias"] = layer.Norm2Weight, layer.Norm2Bias
		out[prefix+".self_attn.in_proj.weight"], out[prefix+".self_attn.out_proj.weight"] = layer.InProjection.Weight, layer.OutProjection.Weight
		out[prefix+".linear1.weight"], out[prefix+".linear2.weight"] = layer.FC1.Weight, layer.FC2.Weight
		if layer.LayerScale1 != nil {
			out[prefix+".layer_scale_1.scale"] = layer.LayerScale1
		}
		if layer.LayerScale2 != nil {
			out[prefix+".layer_scale_2.scale"] = layer.LayerScale2
		}
	}
	return out
}
