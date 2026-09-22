package lfm2

import (
	"math"
	"testing"
)

func tinyLFM2MoESource(cfg Config) fakeLFM2TensorSource {
	prefix := "model.layers.1.feed_forward"
	return fakeLFM2TensorSource{
		prefix + ".gate.weight": {data: []float32{1, 0, 0, 1}, shape: []int{2, 2}},
		prefix + ".expert_bias": {data: []float32{0, 2}, shape: []int{2}},
		prefix + ".experts.gate_up_proj": {data: []float32{
			1, 0, 0, 1, 1, 0, 0, 1,
			2, 0, 0, 2, 1, 0, 0, 1,
		}, shape: []int{2, 4, 2}},
		prefix + ".experts.down_proj": {data: []float32{
			1, 0, 0, 1,
			1, 0, 0, 1,
		}, shape: []int{2, 2, 2}},
	}
}

func TestMoECPURoutingUsesBiasOnlyForSelection(t *testing.T) {
	cfg := tinyLFM2Config(true)
	cfg.NumDenseLayers = 1
	cfg.UseExpertBias = true
	model, err := LoadMoECPU(tinyLFM2MoESource(cfg), cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	selection, err := model.Route([]float32{4, -4})
	if err != nil {
		t.Fatal(err)
	}
	if len(selection.ExpertIDs) != 1 || selection.ExpertIDs[0] != 1 {
		t.Fatalf("selection=%+v", selection)
	}
	wantWeight := float32(1 / (1 + math.Exp(4)))
	if diff := float32(math.Abs(float64(selection.Weights[0] - wantWeight/(wantWeight+1e-6)))); diff > 1e-6 {
		t.Fatalf("weight=%g want=%g", selection.Weights[0], wantWeight/(wantWeight+1e-6))
	}
}

func TestMoECPUForwardTokenMatchesSelectedExpert(t *testing.T) {
	cfg := tinyLFM2Config(true)
	cfg.NumDenseLayers = 1
	cfg.UseExpertBias = true
	model, err := LoadMoECPU(tinyLFM2MoESource(cfg), cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	out, selection, err := model.ForwardToken([]float32{1, 2})
	if err != nil {
		t.Fatal(err)
	}
	if selection.ExpertIDs[0] != 1 || len(out) != 2 || out[0] <= 0 || out[1] <= 0 {
		t.Fatalf("out=%v selection=%+v", out, selection)
	}
}

func TestMoECPURejectsDenseLayerAndNonfiniteLogit(t *testing.T) {
	cfg := tinyLFM2Config(true)
	cfg.NumDenseLayers = 1
	if _, err := LoadMoECPU(tinyLFM2MoESource(cfg), cfg, 0); err == nil {
		t.Fatal("accepted dense layer")
	}
	src := tinyLFM2MoESource(cfg)
	gate := src["model.layers.1.feed_forward.gate.weight"]
	gate.data[0] = float32(math.NaN())
	src["model.layers.1.feed_forward.gate.weight"] = gate
	model, err := LoadMoECPU(src, cfg, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := model.Route([]float32{1, 2}); err == nil {
		t.Fatal("accepted non-finite router logit")
	}
}
