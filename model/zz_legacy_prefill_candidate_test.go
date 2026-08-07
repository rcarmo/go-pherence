package model

import (
	"os"
	"testing"
)

func TestGemma4LegacyPrefillCandidateRealParity(t *testing.T) {
	if os.Getenv("GO_PHERENCE_GEMMA4_PREFILL_CANDIDATE_REAL") == "" {
		t.Skip("set GO_PHERENCE_GEMMA4_PREFILL_CANDIDATE_REAL=1")
	}
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = "../models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf"
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	testGemma4LegacyPrefillCandidateParity(t, m, []int{2, 10979})
}

func TestGemma4LegacyPrefillCandidate124RealParity(t *testing.T) {
	if os.Getenv("GO_PHERENCE_GEMMA4_PREFILL_CANDIDATE_REAL_LONG") == "" {
		t.Skip("set GO_PHERENCE_GEMMA4_PREFILL_CANDIDATE_REAL_LONG=1")
	}
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = "../models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf"
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	prepared := make([]int, 124)
	prepared[0] = 2
	for i := 1; i < len(prepared); i++ {
		prepared[i] = 10979
	}
	testGemma4LegacyPrefillCandidateParity(t, m, prepared)
}

func testGemma4LegacyPrefillCandidateParity(t *testing.T, m *LlamaModel, prepared []int) {
	t.Helper()
	legacy, err := newCPUTokenStateForLegacyGenerate(m, prepared, 2)
	if err != nil {
		t.Fatal(err)
	}
	var legacyToken int
	var legacyLogits []float32
	for pos, tok := range prepared {
		next, logits, _, err := m.runLegacyCPUToken(legacy, tok, pos, nil)
		if err != nil {
			t.Fatal(err)
		}
		legacyToken, legacyLogits = next, logits
	}
	legacyHidden := append([]float32(nil), legacy.hidden...)

	candidate, err := newCPUTokenStateForLegacyGenerate(m, prepared, 2)
	if err != nil {
		t.Fatal(err)
	}
	hiddenRows, err := m.runLegacyCPUPrefillBatchLayers(candidate, prepared, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(hiddenRows) != len(prepared) {
		t.Fatalf("candidate rows=%d want=%d", len(hiddenRows), len(prepared))
	}
	candidateHidden := hiddenRows[len(hiddenRows)-1]
	_, candidateLogits, candidateToken, err := m.finishCPUDecodeStep(candidateHidden)
	if err != nil {
		t.Fatal(err)
	}
	if candidateToken != legacyToken {
		t.Fatalf("candidate token=%d legacy=%d", candidateToken, legacyToken)
	}
	for i := range legacyHidden {
		if candidateHidden[i] != legacyHidden[i] {
			t.Fatalf("final activation differs at %d: candidate=%g legacy=%g", i, candidateHidden[i], legacyHidden[i])
		}
	}
	for i := range legacyLogits {
		if candidateLogits[i] != legacyLogits[i] {
			t.Fatalf("logit %d candidate=%g legacy=%g", i, candidateLogits[i], legacyLogits[i])
		}
	}
	for layer := range legacy.kvCacheK {
		for _, caches := range [][2][]float32{{legacy.kvCacheK[layer], candidate.kvCacheK[layer]}, {legacy.kvCacheV[layer], candidate.kvCacheV[layer]}} {
			if len(caches[0]) != len(caches[1]) {
				t.Fatalf("layer %d KV length candidate=%d legacy=%d", layer, len(caches[1]), len(caches[0]))
			}
			for i := range caches[0] {
				if caches[1][i] != caches[0][i] {
					t.Fatalf("layer %d KV differs at %d: candidate=%g legacy=%g", layer, i, caches[1][i], caches[0][i])
				}
			}
		}
	}
}
