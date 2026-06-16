package model

import (
	"fmt"

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

func validateGGUFMatrixDims(name string, w *gguf.QuantMatrix, outDim, inDim int) error {
	if w == nil {
		return nil
	}
	if w.OutDim != outDim || w.InDim != inDim {
		return fmt.Errorf("%s GGUF dims out/in=%d/%d, want %d/%d", name, w.OutDim, w.InDim, outDim, inDim)
	}
	return nil
}
