package model

import "testing"

func TestGGUFGenerateWithOptionsCreatesCompressedKV(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 4, MaxSeqLen: 4, VocabSize: 2}}
	caches, err := m.NewTurboQuantKVCache("turbo4", "turbo2", 1)
	if err != nil {
		t.Fatal(err)
	}
	st := m.NewForwardState()
	st.compressedKV = caches
	if len(st.compressedKV) != 2 || st.compressedKV[0] == nil {
		t.Fatalf("bad compressed kv state: %+v", st.compressedKV)
	}
}

func TestGGUFGenerateWithOptionsValidatesTurboQuantPolicy(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 1, MaxSeqLen: 4, VocabSize: 2}}
	if _, err := m.GenerateWithOptions([]int{0}, 1, GGUFGenerationOptions{CacheTypeK: "turbo9", KVResidualWindow: -1}); err == nil {
		t.Fatal("expected invalid cache policy error")
	}
}

func TestGGUFGenerateHandlesEmptyAndDoesNotDoublePrefill(t *testing.T) {
	m := &GGUFLlama{Config: GGUFLlamaConfig{NumLayers: 0, NumKVHeads: 1, HeadDim: 1, MaxSeqLen: 4, VocabSize: 2}}
	if got, err := m.Generate(nil, 1); err != nil || len(got) != 0 {
		t.Fatalf("empty prompt got=%v err=%v", got, err)
	}
	if got, err := m.Generate([]int{0}, 0); err != nil || len(got) != 0 {
		t.Fatalf("zero max got=%v err=%v", got, err)
	}
}
