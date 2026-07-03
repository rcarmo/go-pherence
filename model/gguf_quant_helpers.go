package model

import (
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func gemvGGUFTo(out, x []float32, w *gguf.QuantMatrix, inDim, outDim int) bool {
	if w == nil {
		return false
	}
	if err := gemvGGUFQuantRows(out, x, w, inDim, outDim); err != nil {
		return false
	}
	return true
}
