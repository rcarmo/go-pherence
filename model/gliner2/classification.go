package gliner2

import (
	"fmt"
	"math"
)

// ClassificationHead scores independently encoded choice-marker states using
// the published ReLU MLP, not the GELU used by the span reranker.
type ClassificationHead struct{ Input, Output Linear }

func LoadClassificationHead(source TensorSource, hidden int) (ClassificationHead, error) {
	if source == nil || hidden <= 0 {
		return ClassificationHead{}, fmt.Errorf("source and hidden width required")
	}
	r := weightReader{source: source}
	h := ClassificationHead{r.linear("classifier.0", hidden, 2*hidden), r.linear("classifier.3", 2*hidden, 1)}
	if r.err != nil {
		return ClassificationHead{}, r.err
	}
	return h, h.Validate()
}
func (h ClassificationHead) Validate() error {
	if err := h.Input.Validate(); err != nil {
		return err
	}
	if err := h.Output.Validate(); err != nil {
		return err
	}
	if h.Output.InDim != h.Input.OutDim || h.Output.OutDim != 1 {
		return fmt.Errorf("classifier projection mismatch")
	}
	return nil
}
func (h ClassificationHead) Forward(choices [][]float32) ([]float32, error) {
	if err := h.Validate(); err != nil {
		return nil, err
	}
	flat, err := flattenRows(choices, h.Input.InDim, "classification choices")
	if err != nil {
		return nil, err
	}
	mid := make([]float32, len(choices)*h.Input.OutDim)
	if err = h.Input.ApplyBatch(flat, mid, len(choices)); err != nil {
		return nil, err
	}
	for i, v := range mid {
		if v < 0 {
			mid[i] = 0
		}
	}
	out := make([]float32, len(choices))
	if err = h.Output.ApplyBatch(mid, out, len(choices)); err != nil {
		return nil, err
	}
	return out, nil
}

// Probabilities preserves independent sigmoid scores; values need not sum to 1.
func (h ClassificationHead) Probabilities(choices [][]float32, temperature float64) ([]float64, error) {
	if temperature <= 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
		return nil, fmt.Errorf("invalid classification temperature")
	}
	logits, err := h.Forward(choices)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = sigmoid(float64(v) / temperature)
	}
	return out, nil
}
