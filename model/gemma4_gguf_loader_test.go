package model

import (
	"os"
	"regexp"
	"sort"
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func openLocalGemma4GGUFForTest(t *testing.T) *gguf.GGUF {
	t.Helper()
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = "models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		if _, parentErr := os.Stat("../" + path); parentErr == nil {
			path = "../" + path
		} else {
			t.Skipf("Gemma4 GGUF unavailable: %v", err)
		}
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestGemma4GGUFLayerOutputScaleIsScalarF32(t *testing.T) {
	g := openLocalGemma4GGUFForTest(t)
	defer g.Close()
	for l := 0; ; l++ {
		name := "blk." + itoa(l) + ".layer_output_scale.weight"
		tensor, ok := g.TensorByName(name)
		if !ok {
			if l == 0 {
				t.Fatal("missing blk.0.layer_output_scale.weight")
			}
			return
		}
		if tensor.QType != gguf.QuantF32 || gguf.TensorElements(tensor.Shape) != 1 {
			t.Fatalf("%s qtype/shape=%s/%v, want F32/[1]", name, tensor.QType, tensor.Shape)
		}
	}
}

func TestGemma4GGUFTensorPatternCoverage(t *testing.T) {
	g := openLocalGemma4GGUFForTest(t)
	defer g.Close()
	want := map[string]int{
		"blk.*.attn_k.weight":              24,
		"blk.*.attn_k_norm.weight":         24,
		"blk.*.attn_norm.weight":           42,
		"blk.*.attn_output.weight":         42,
		"blk.*.attn_q.weight":              42,
		"blk.*.attn_q_norm.weight":         42,
		"blk.*.attn_v.weight":              24,
		"blk.*.ffn_down.weight":            42,
		"blk.*.ffn_gate.weight":            42,
		"blk.*.ffn_norm.weight":            42,
		"blk.*.ffn_up.weight":              42,
		"blk.*.inp_gate.weight":            42,
		"blk.*.layer_output_scale.weight":  42,
		"blk.*.post_attention_norm.weight": 42,
		"blk.*.post_ffw_norm.weight":       42,
		"blk.*.post_norm.weight":           42,
		"blk.*.proj.weight":                42,
		"output_norm.weight":               1,
		"per_layer_model_proj.weight":      1,
		"per_layer_proj_norm.weight":       1,
		"per_layer_token_embd.weight":      1,
		"rope_freqs.weight":                1,
		"token_embd.weight":                1,
	}
	re := regexp.MustCompile(`^blk\.[0-9]+\.`)
	got := map[string]int{}
	for _, tensor := range g.Tensors {
		got[re.ReplaceAllString(tensor.Name, "blk.*.")]++
	}
	if len(got) != len(want) {
		t.Fatalf("Gemma4 tensor pattern count=%d want %d got=%v", len(got), len(want), sortedPatternCounts(got))
	}
	for pattern, wantCount := range want {
		if got[pattern] != wantCount {
			t.Fatalf("Gemma4 tensor pattern %s count=%d want %d; all=%v", pattern, got[pattern], wantCount, sortedPatternCounts(got))
		}
	}
}

func TestGemma4GGUFSharedKVSourceMapping(t *testing.T) {
	g := openLocalGemma4GGUFForTest(t)
	defer g.Close()
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = "models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf"
		if _, err := os.Stat(path); err != nil {
			if _, parentErr := os.Stat("../" + path); parentErr == nil {
				path = "../" + path
			}
		}
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	if m.Config.NumKVSharedLayers != 18 || m.Config.NumLayers != 42 {
		t.Fatalf("layers/shared=%d/%d want 42/18", m.Config.NumLayers, m.Config.NumKVSharedLayers)
	}
	for _, l := range []int{0, 22, 23} {
		if !m.Layers[l].HasKV {
			t.Fatalf("layer %d should own KV", l)
		}
	}
	for _, l := range []int{24, 25, 40} {
		if m.Layers[l].HasKV || m.Layers[l].KVSourceLayer != 22 {
			t.Fatalf("sliding shared layer %d hasKV/src=%v/%d want false/22", l, m.Layers[l].HasKV, m.Layers[l].KVSourceLayer)
		}
	}
	for _, l := range []int{29, 35, 41} {
		if m.Layers[l].HasKV || m.Layers[l].KVSourceLayer != 23 {
			t.Fatalf("full shared layer %d hasKV/src=%v/%d want false/23", l, m.Layers[l].HasKV, m.Layers[l].KVSourceLayer)
		}
	}
	if m.LMHeadGGUF != m.EmbedTokensGGUF || len(m.EmbedTokensGGUF.Raw) == 0 || m.EmbedTokensGGUF.UsesLlamaQ4_0x8() {
		t.Fatalf("tied embedding/head ownership changed: tied=%v raw=%d packed=%v", m.LMHeadGGUF == m.EmbedTokensGGUF, len(m.EmbedTokensGGUF.Raw), m.EmbedTokensGGUF.UsesLlamaQ4_0x8())
	}
	if m.Layers[0].QWGGUF.UsesLlamaQ4_0x8() {
		for i, layer := range m.Layers {
			projections := []*gguf.QuantMatrix{layer.QWGGUF, layer.OWGGUF, layer.GateWGGUF, layer.UpWGGUF, layer.DownWGGUF, layer.PLIGateGGUF, layer.PLIProjGGUF}
			if layer.HasKV {
				projections = append(projections, layer.KWGGUF, layer.VWGGUF)
			}
			for _, projection := range projections {
				if projection != nil && projection.QType == gguf.QuantQ4_0 && (!projection.UsesLlamaQ4_0x8() || projection.Raw != nil) {
					t.Fatalf("layer %d projection %s ownership raw=%d packed=%v", i, projection.Name, len(projection.Raw), projection.UsesLlamaQ4_0x8())
				}
			}
		}
	}
}

func TestGemma4GGUFPerLayerInputTensorShapes(t *testing.T) {
	g := openLocalGemma4GGUFForTest(t)
	defer g.Close()
	cfg, err := gemma4GGUFConfig(g)
	if err != nil {
		t.Fatal(err)
	}
	totalPerLayerDim := cfg.NumLayers * cfg.HiddenPerLayer
	checks := []struct {
		name  string
		qtype gguf.QuantType
		shape []uint64
	}{
		{"per_layer_model_proj.weight", gguf.QuantF16, []uint64{uint64(cfg.HiddenSize), uint64(totalPerLayerDim)}},
		{"per_layer_proj_norm.weight", gguf.QuantF32, []uint64{uint64(cfg.HiddenPerLayer)}},
		{"per_layer_token_embd.weight", gguf.QuantQ6_K, []uint64{uint64(totalPerLayerDim), uint64(cfg.VocabPerLayer)}},
	}
	for _, check := range checks {
		tensor, ok := g.TensorByName(check.name)
		if !ok {
			t.Fatalf("missing %s", check.name)
		}
		if tensor.QType != check.qtype || !sameUint64s(tensor.Shape, check.shape) {
			t.Fatalf("%s qtype/shape=%s/%v, want %s/%v", check.name, tensor.QType, tensor.Shape, check.qtype, check.shape)
		}
	}
}

func sortedPatternCounts(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k, v := range m {
		out = append(out, k+"="+itoa(v))
	}
	sort.Strings(out)
	return out
}

func sameUint64s(a, b []uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
