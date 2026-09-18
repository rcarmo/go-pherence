package gliner2

import (
	"math"
	"testing"
)

func TestClassificationReLUAndIndependentSigmoid(t *testing.T) {
	h := ClassificationHead{Input: Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}}, Output: Linear{InDim: 2, OutDim: 1, Weight: []float32{2, 3}, Bias: []float32{1}}}
	rows := [][]float32{{-1, 2}, {1, -2}}
	got, err := h.Forward(rows)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 7 || got[1] != 3 {
		t.Fatal(got)
	}
	probs, err := h.Probabilities(rows, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(probs[0]-sigmoid(3.5)) > 1e-8 || probs[0]+probs[1] <= 1 {
		t.Fatal(probs)
	}
	if _, err = h.Probabilities(rows, 0); err == nil {
		t.Fatal("bad temperature accepted")
	}
}
func TestClassifierPublishedShapeBinding(t *testing.T) {
	s := shapeSource{"classifier.0.weight": {1536, 768}, "classifier.0.bias": {1536}, "classifier.3.weight": {1, 1536}, "classifier.3.bias": {1}}
	if _, err := LoadClassificationHead(s, 768); err != nil {
		t.Fatal(err)
	}
	delete(s, "classifier.3.bias")
	if _, err := LoadClassificationHead(s, 768); err == nil {
		t.Fatal("missing bias accepted")
	}
}
