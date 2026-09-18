package model

import (
	"strings"
	"testing"
)

func encodeTokenHiddenStatesSequentialReference(t *testing.T, m *LlamaModel, tokenIDs []int) [][]float32 {
	t.Helper()
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	hiddenRows := make([][]float32, len(tokenIDs))
	h := m.Config.HiddenSize
	for pos, tokID := range tokenIDs {
		hidden := make([]float32, h)
		if err := m.ScaledTokenEmbeddingInto(hidden, tokID); err != nil {
			t.Fatalf("token %d embedding: %v", pos, err)
		}
		for layerIdx := 0; layerIdx < m.Config.NumLayers; layerIdx++ {
			hidden = m.ForwardLayer(hidden, layerIdx, pos, pos, kvCacheK, kvCacheV)
			if hidden == nil {
				t.Fatalf("token %d layer %d forward failed", pos, layerIdx)
			}
		}
		act, err := m.FinishCPUActivation(hidden)
		if err != nil {
			t.Fatalf("token %d activation: %v", pos, err)
		}
		hiddenRows[pos] = act
	}
	return hiddenRows
}

func TestEncodeTokenHiddenStatesMatchesPrefillAndSequential(t *testing.T) {
	t.Setenv("GO_PHERENCE_DISABLE_CPU_PREFILL", "0")
	m := buildPrefillTestModel("qwen3", false, true, false)
	tokenIDs := []int{1, 5, 9, 2, 7, 3}
	if !m.prefillCPUEligible(len(tokenIDs)) {
		t.Fatal("model expected to be prefill-eligible")
	}
	got, err := m.EncodeTokenHiddenStates(tokenIDs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(tokenIDs) {
		t.Fatalf("hidden rows=%d want %d", len(got), len(tokenIDs))
	}
	wantSeq := encodeTokenHiddenStatesSequentialReference(t, m, tokenIDs)
	for i := range wantSeq {
		if !sameFloat32s(got[i], wantSeq[i]) {
			t.Fatalf("row %d=%v want %v", i, got[i], wantSeq[i])
		}
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	lastHidden, ok := m.prefillCPU(tokenIDs, kvCacheK, kvCacheV)
	if !ok {
		t.Fatal("prefillCPU unexpectedly fell back")
	}
	wantLast, err := m.FinishCPUActivation(lastHidden)
	if err != nil {
		t.Fatal(err)
	}
	if !sameFloat32s(got[len(got)-1], wantLast) {
		t.Fatalf("last row=%v want %v", got[len(got)-1], wantLast)
	}
}

func TestEncodeTokenHiddenStatesSupportsSingletonSequential(t *testing.T) {
	m := buildPrefillTestModel("llama", false, false, false)
	tokenIDs := []int{7}
	if m.prefillCPUEligible(len(tokenIDs)) {
		t.Fatal("singleton prompt should not be prefill-eligible")
	}
	got, err := m.EncodeTokenHiddenStates(tokenIDs)
	if err != nil {
		t.Fatal(err)
	}
	want := encodeTokenHiddenStatesSequentialReference(t, m, tokenIDs)
	if len(got) != 1 || !sameFloat32s(got[0], want[0]) {
		t.Fatalf("singleton rows=%v want %v", got, want)
	}
}

func TestEncodeTokenHiddenStatesRejectsInvalidInput(t *testing.T) {
	t.Run("nil model", func(t *testing.T) {
		if _, err := (*LlamaModel)(nil).EncodeTokenHiddenStates([]int{0}); err == nil {
			t.Fatal("accepted nil model")
		}
	})
	t.Run("empty", func(t *testing.T) {
		m := buildPrefillTestModel("llama", false, false, false)
		if _, err := m.EncodeTokenHiddenStates(nil); err == nil {
			t.Fatal("accepted empty token sequence")
		}
	})
	t.Run("bad token", func(t *testing.T) {
		m := buildPrefillTestModel("llama", false, false, false)
		if _, err := m.EncodeTokenHiddenStates([]int{-1}); err == nil {
			t.Fatal("accepted negative token id")
		}
	})
	t.Run("gemma4 unsupported", func(t *testing.T) {
		m := buildPrefillTestModel("gemma4_text", false, false, false)
		if _, err := m.EncodeTokenHiddenStates([]int{1}); err == nil || !strings.Contains(strings.ToLower(err.Error()), "gemma4") {
			t.Fatalf("gemma4 error=%v", err)
		}
	})
	t.Run("moe unsupported", func(t *testing.T) {
		m := buildPrefillTestModel("llama", false, false, false)
		m.Layers[0].IsMoE = true
		if _, err := m.EncodeTokenHiddenStates([]int{1}); err == nil || !strings.Contains(strings.ToLower(err.Error()), "mixture-of-experts") {
			t.Fatalf("moe error=%v", err)
		}
	})
}
