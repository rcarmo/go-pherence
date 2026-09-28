package lfm2

import (
	"math"
	"strconv"
	"strings"
	"testing"
)

func syntheticLFM2DecoderSource(cfg Config) fakeLFM2TensorSource {
	s := fakeLFM2TensorSource{
		"model.embed_tokens.weight":   {data: []float32{0, 0, 1, 0, 0, 1, 1, 1}, shape: []int{4, 2}},
		"model.embedding_norm.weight": {data: []float32{1, 1}, shape: []int{2}},
	}
	for name, tensor := range tinyLFM2ConvSource(cfg) {
		s[name] = tensor
	}
	for name, tensor := range tinyLFM2AttentionSource(cfg) {
		s[name] = tensor
	}
	for name, tensor := range tinyLFM2MoESource(cfg) {
		s[name] = tensor
	}
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		prefix := "model.layers." + strconv.Itoa(layer)
		s[prefix+".operator_norm.weight"] = struct {
			data  []float32
			shape []int
		}{[]float32{1, 1}, []int{2}}
		s[prefix+".ffn_norm.weight"] = struct {
			data  []float32
			shape []int
		}{[]float32{1, 1}, []int{2}}
	}
	// Make operators and routed experts no-ops so residual ownership and greedy
	// generation have an exact, easy-to-audit oracle.
	for name, tensor := range s {
		if name == "model.embed_tokens.weight" || name == "model.embedding_norm.weight" || strings.HasSuffix(name, "norm.weight") {
			continue
		}
		clear(tensor.data)
		s[name] = tensor
	}
	for _, suffix := range []string{"w1", "w3"} {
		s["model.layers.0.feed_forward."+suffix+".weight"] = struct {
			data  []float32
			shape []int
		}{make([]float32, cfg.IntermediateSize*cfg.HiddenSize), []int{cfg.IntermediateSize, cfg.HiddenSize}}
	}
	s["model.layers.0.feed_forward.w2.weight"] = struct {
		data  []float32
		shape []int
	}{make([]float32, cfg.HiddenSize*cfg.IntermediateSize), []int{cfg.HiddenSize, cfg.IntermediateSize}}
	return s
}

func TestDecoderCPUForwardAndGreedyGeneration(t *testing.T) {
	cfg := tinyLFM2Config(true)
	model, err := LoadDecoderCPU(syntheticLFM2DecoderSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	state := model.NewState()
	hidden, next, err := model.ForwardToken(1, state)
	if err != nil {
		t.Fatal(err)
	}
	if len(hidden) != cfg.HiddenSize || next.Pos != 1 || len(next.Conv[0]) != cfg.HiddenSize*cfg.ConvLCache || len(next.Attention[1].K) != cfg.NumKeyValueHeads*cfg.HeadDim {
		t.Fatalf("hidden/state=%v %+v", hidden, next)
	}
	if state.Pos != 0 || len(state.Attention[1].K) != 0 {
		t.Fatalf("caller state mutated: %+v", state)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Tokens: []uint32{1}, MaxNewTokens: 2, BytesPerFloat: 4})
	if err != nil {
		t.Fatal(err)
	}
	generated, err := model.Generate(plan)
	if err != nil {
		t.Fatal(err)
	}
	if len(generated) != 2 || generated[0] != 1 || generated[1] != 1 {
		t.Fatalf("generated=%v", generated)
	}
}

func TestDecoderCPURejectsMalformedStateAndLogits(t *testing.T) {
	cfg := tinyLFM2Config(true)
	model, err := LoadDecoderCPU(syntheticLFM2DecoderSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := model.ForwardToken(uint32(cfg.VocabSize), model.NewState()); err == nil {
		t.Fatal("accepted out-of-range token")
	}
	state := model.NewState()
	state.Conv = state.Conv[:1]
	if _, _, err := model.ForwardToken(1, state); err == nil {
		t.Fatal("accepted malformed state")
	}
	if _, err := greedyLFM2Token([]float32{0, float32(math.NaN())}); err == nil {
		t.Fatal("accepted NaN logit")
	}
}
