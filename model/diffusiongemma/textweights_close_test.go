package diffusiongemma

import "testing"

func TestTextWeightsCloseClearsCachesAndGGUFReferences(t *testing.T) {
	w := &TextWeights{
		floatCache: map[string]FloatTensor{
			"x": {Data: []float32{1}, Shape: []int{1}, DType: "F32"},
		},
		ggufTokenEmbd: nil,
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	if w.floatCache != nil {
		t.Fatalf("floatCache not cleared")
	}
	if w.ggufTokenEmbd != nil {
		t.Fatalf("ggufTokenEmbd not cleared")
	}
}
