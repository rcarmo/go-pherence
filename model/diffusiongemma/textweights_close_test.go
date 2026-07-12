package diffusiongemma

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestTextWeightsCloseClearsCachesAndGGUFReferences(t *testing.T) {
	w := &TextWeights{
		floatCache: map[string]FloatTensor{
			"x": {Data: []float32{1}, Shape: []int{1}, DType: "F32"},
		},
		ggufQuant: map[string]*gguf.QuantMatrix{
			"q": {Name: "q", QType: gguf.QuantQ8_0, InDim: 32, OutDim: 1},
		},
		ggufTokenEmbd: nil,
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if w.floatCache != nil {
		t.Fatalf("floatCache not cleared")
	}
	if w.ggufQuant != nil {
		t.Fatalf("ggufQuant not cleared")
	}
	if w.ggufTokenEmbd != nil {
		t.Fatalf("ggufTokenEmbd not cleared")
	}
}
