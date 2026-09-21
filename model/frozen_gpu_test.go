package model

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func frozenGPUValidConfig() LlamaConfig {
	return LlamaConfig{ModelType: "qwen3", VocabSize: 32, HiddenSize: 16, Intermediate: 32, NumLayers: 2, NumHeads: 2, NumKVHeads: 1, HeadDim: 8, MaxSeqLen: 512, RMSNormEps: 1e-6, RopeTheta: 10000, HiddenAct: "silu"}
}
func TestFrozenGPUAdmission(t *testing.T) {
	cfg := frozenGPUValidConfig()
	if err := validateFrozenGPUConfig(cfg, 128); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*LlamaConfig){func(c *LlamaConfig) { c.ModelType = "qwen3_moe" }, func(c *LlamaConfig) { c.SlidingWindow = 128 }, func(c *LlamaConfig) { c.NumExperts = 8 }, func(c *LlamaConfig) { c.RMSNormEps = math.NaN() }, func(c *LlamaConfig) { c.RopeTheta = 0 }, func(c *LlamaConfig) { c.HeadDim = 7 }, func(c *LlamaConfig) { c.NumLayers = 1 << 20 }} {
		bad := cfg
		mutate(&bad)
		if validateFrozenGPUConfig(bad, 128) == nil {
			t.Fatal("accepted invalid config", bad)
		}
	}
	for _, ids := range [][]int{nil, {-1}, {32}, make([]int, 129)} {
		if validateFrozenGPURequest(cfg, 128, ids) == nil {
			t.Fatal("accepted ids", len(ids))
		}
	}
	if _, err := frozenGPUResolveMaxTokens(cfg, FrozenGPUOptions{MaxTokens: -1}); err == nil {
		t.Fatal("negative limit")
	}
	if _, err := frozenGPUResolveMaxTokens(cfg, FrozenGPUOptions{MaxTokens: 513}); err == nil {
		t.Fatal("unbounded limit")
	}
	if checkFrozenGPUBudget(100, 200, FrozenGPUOptions{BudgetBytes: 100, ReserveBytes: 100}) != nil {
		t.Fatal("exact fit rejected")
	}
	for _, o := range []FrozenGPUOptions{{BudgetBytes: 99, ReserveBytes: 1}, {BudgetBytes: 200, ReserveBytes: 101}, {BudgetBytes: 200, ReserveBytes: 200}} {
		if checkFrozenGPUBudget(100, 200, o) == nil {
			t.Fatal("budget accepted", o)
		}
	}
	if checkFrozenGPUBudget(100, 0, FrozenGPUOptions{}) == nil {
		t.Fatal("unknown free memory accepted")
	}
	if _, err := NewFrozenGPUEncoder("unused", FrozenGPUOptions{}); err == nil {
		t.Fatal("implicit budget accepted")
	}
	var nilEncoder *FrozenGPUEncoder
	if _, err := nilEncoder.EncodeTokenHiddenStates([]int{0}); err == nil {
		t.Fatal("nil encoder accepted")
	}
	nilEncoder.Close()
}

func TestFrozenGPURejectsUnsupportedSidecarBeforeCUDA(t *testing.T) {
	for _, field := range []string{`"attention_bias":true`, `"use_sliding_window":true`, `"rope_scaling":{"factor":2}`, `"quantization_config":{"bits":4}`} {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{"model_type":"qwen3",`+field+`}`), 0o600); err != nil {
			t.Fatal(err)
		}
		if _, err := readFrozenGPUConfig(dir); err == nil {
			t.Fatal("accepted unsupported policy", field)
		}
	}
}

func TestFrozenGPUMemoryPlanScalesWithPromptNotLayerKV(t *testing.T) {
	cfg := frozenGPUValidConfig()
	host := &frozenGPUHostModel{matrixBytes: 1024, normBytes: 256, maxMatrixElem: 512}
	a, err := estimateFrozenGPUBytes(cfg, host, 1)
	if err != nil {
		t.Fatal(err)
	}
	b, err := estimateFrozenGPUBytes(cfg, host, 128)
	if err != nil {
		t.Fatal(err)
	}
	if a.matrix != b.matrix || a.kv != 0 || b.kv != 0 || b.scratch <= a.scratch {
		t.Fatal(a, b)
	}
	if b.total != b.matrix+b.norm+b.rope+b.scratch {
		t.Fatal("budget components inconsistent")
	}
}
