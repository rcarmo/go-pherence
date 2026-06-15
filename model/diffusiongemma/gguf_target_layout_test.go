package diffusiongemma

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestLocalDiffusionGemmaGGUFQ4KMTargetLayout(t *testing.T) {
	path := filepath.Join("..", "..", "..", "llama.cpp", "models", "diffusiongemma-gguf", "diffusiongemma-26B-A4B-it-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		t.Skip("local DiffusionGemma GGUF Q4_K_M reference not present")
	}
	g, err := gguf.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if _, ok := g.TensorByName("output.weight"); ok {
		t.Fatalf("DiffusionGemma GGUF target unexpectedly has separate output.weight; tied token_embd.weight is expected")
	}
	tok, ok := g.TensorByName("token_embd.weight")
	if !ok {
		t.Fatalf("token_embd.weight missing")
	}
	if tok.QType != gguf.QuantQ6_K || len(tok.Shape) != 2 || tok.Shape[0] != 2816 || tok.Shape[1] != 262144 {
		t.Fatalf("token_embd.weight qtype/shape=%s/%v, want Q6_K/[2816 262144]", tok.QType, tok.Shape)
	}
	for _, tensor := range g.Tensors {
		name := tensor.Name
		if strings.Contains(name, "vision") || strings.Contains(name, "patch") || strings.Contains(name, "image") || strings.Contains(name, "mm") {
			t.Fatalf("DiffusionGemma GGUF Q4_K_M target unexpectedly has vision-like tensor %q shape=%v qtype=%s; current target is text-only", name, tensor.Shape, tensor.QType)
		}
	}
	gate, ok := g.TensorByName("blk.0.ffn_gate_up_exps.weight")
	if !ok || gate.QType != gguf.QuantQ4_K {
		t.Fatalf("blk.0 gate_up experts qtype=%v ok=%v, want Q4_K", gate.QType, ok)
	}
	down, ok := g.TensorByName("blk.0.ffn_down_exps.weight")
	if !ok || down.QType != gguf.QuantQ8_0 {
		t.Fatalf("blk.0 down experts qtype=%v ok=%v, want Q8_0", down.QType, ok)
	}
	meta, err := LoadMetadata(filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8"))
	if err != nil {
		t.Fatal(err)
	}
	weights, err := OpenTextWeightsFromGGUF(g, meta.Shape)
	if err != nil {
		t.Fatal(err)
	}
	defer weights.Close()
	if weights.ggufTokenEmbd == nil || weights.ggufTokenEmbd.QType != gguf.QuantQ6_K || weights.ggufTokenEmbd.OutDim != 262144 || weights.ggufTokenEmbd.InDim != 2816 {
		t.Fatalf("ggufTokenEmbd=%+v, want Q6_K [in=2816,out=262144]", weights.ggufTokenEmbd)
	}
}
