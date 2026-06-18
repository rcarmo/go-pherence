package model

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestGemma4GGUFLayerOutputScaleIsScalarF32(t *testing.T) {
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
