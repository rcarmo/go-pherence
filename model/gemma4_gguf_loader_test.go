package model

import (
	"os"
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
