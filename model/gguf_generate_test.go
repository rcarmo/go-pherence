package model

import "testing"

func TestGGUFGenerateHandlesEmptyAndDoesNotDoublePrefill(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 0, NumKVHeads: 1, HeadDim: 1, MaxSeqLen: 4, VocabSize: 2}}
	if got, err := m.Generate(nil, 1); err != nil || len(got) != 0 {
		t.Fatalf("empty prompt got=%v err=%v", got, err)
	}
	if got, err := m.Generate([]int{0}, 0); err != nil || len(got) != 0 {
		t.Fatalf("zero max got=%v err=%v", got, err)
	}
}
