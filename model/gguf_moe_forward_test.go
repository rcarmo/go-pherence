package model

import "testing"

func TestGGUFTopKExpertsAppliesREAPMask(t *testing.T) {
	reap := &REAPConfig{Enabled: true, LayerActiveNumeric: map[int]map[int]bool{2: {1: true, 3: true}}}
	got := ggufTopKExperts([]float32{0.9, 0.2, 0.8, 0.7}, 2, reap, 2)
	if len(got) != 2 || got[0].id != 3 || got[1].id != 1 {
		t.Fatalf("selected=%+v", got)
	}
}

func TestGGUFSharedExpertAddSkipsMissingWeights(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{HiddenSize: 4, MoEHiddenSize: 4, SharedMoEHiddenSize: 4}}
	out := []float32{0, 0, 0, 0}
	m.ggufSharedExpertAdd(out, make([]float32, 4), make([]float32, 4), make([]float32, 4), make([]float32, 4), &GGUFLlamaLayer{})
	for i, v := range out {
		if v != 0 {
			t.Fatalf("out[%d]=%v", i, v)
		}
	}
}

func TestGGUFMoEForwardRejectsIncompleteLayer(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{HiddenSize: 4, NumExperts: 2, NumExpertsPerTok: 1, MoEHiddenSize: 4}}
	out := []float32{1, 1, 1, 1}
	m.ggufMoEForward(out, make([]float32, 4), make([]float32, 4), make([]float32, 4), make([]float32, 4), &GGUFLlamaLayer{}, 0)
	for i, v := range out {
		if v != 0 {
			t.Fatalf("out[%d]=%v want zero", i, v)
		}
	}
}
