package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/loader/gguf"
)

func TestGemvGGUFQuantRowsRejectsBadDims(t *testing.T) {
	w := &gguf.QuantMatrix{InDim: 2, OutDim: 2}
	if err := gemvGGUFQuantRows(make([]float32, 1), []float32{1, 2}, w, 2, 2); err == nil {
		t.Fatal("expected bad output size error")
	}
	if err := gemvGGMLQuantRows(make([]float32, 2), []float32{1, 2}, w, 2, 2); err == nil {
		t.Fatal("expected unavailable ggml quant rows without quant type/raw data")
	}
}
