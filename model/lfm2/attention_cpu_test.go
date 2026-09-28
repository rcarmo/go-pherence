package lfm2

import "testing"

func tinyLFM2AttentionSource(cfg Config) fakeLFM2TensorSource {
	identity := []float32{1, 0, 0, 1}
	prefix := "model.layers.1.self_attn."
	return fakeLFM2TensorSource{
		prefix + "q_proj.weight":      {data: identity, shape: []int{2, 2}},
		prefix + "k_proj.weight":      {data: identity, shape: []int{2, 2}},
		prefix + "v_proj.weight":      {data: identity, shape: []int{2, 2}},
		prefix + "out_proj.weight":    {data: identity, shape: []int{2, 2}},
		prefix + "q_layernorm.weight": {data: []float32{1, 1}, shape: []int{2}},
		prefix + "k_layernorm.weight": {data: []float32{1, 1}, shape: []int{2}},
	}
}

func TestFullAttentionCPUForwardTokenOwnsKV(t *testing.T) {
	cfg := tinyLFM2Config(true)
	model, err := LoadFullAttentionCPU(tinyLFM2AttentionSource(cfg), cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	out0, state0, err := model.ForwardToken([]float32{1, 0}, AttentionCPUState{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(out0) != 2 || out0[0] == 0 || len(state0.K) != 2 || len(state0.V) != 2 {
		t.Fatalf("first out/state=%v %+v", out0, state0)
	}
	frozen := state0.Clone()
	out1, state1, err := model.ForwardToken([]float32{0, 1}, state0, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(out1) != 2 || len(state1.K) != 4 || len(state1.V) != 4 {
		t.Fatalf("second out/state=%v %+v", out1, state1)
	}
	for i := range frozen.K {
		if state0.K[i] != frozen.K[i] || state0.V[i] != frozen.V[i] {
			t.Fatalf("caller state mutated: before=%+v after=%+v", frozen, state0)
		}
	}
}

func TestFullAttentionCPURejectsLayerPositionAndState(t *testing.T) {
	cfg := tinyLFM2Config(true)
	src := tinyLFM2AttentionSource(cfg)
	if _, err := LoadFullAttentionCPU(src, cfg, 0); err == nil {
		t.Fatal("accepted convolution layer")
	}
	model, err := LoadFullAttentionCPU(src, cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := model.ForwardToken([]float32{1, 2}, AttentionCPUState{}, -1); err == nil {
		t.Fatal("accepted negative position")
	}
	if _, _, err := model.ForwardToken([]float32{1, 2}, AttentionCPUState{K: []float32{1}}, 1); err == nil {
		t.Fatal("accepted malformed state")
	}
}
