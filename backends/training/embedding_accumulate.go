package training

import (
	"context"
	"fmt"
)

// CompactEmbeddingAccumulateShape bounds compact VJP to dense resident
// accumulation: output=previous+scale*compact(slot), absent tokens use +0.
// Scale is applied once after each sequence-ordered token sum. The compact
// index must already pass ValidateCompactEmbeddingIndex before dispatch.
func CompactEmbeddingAccumulateShape(rows, width, vocabulary int, scale float32) (dense, compact, index int, err error) {
	if _, _, err = EmbeddingShape(rows, width, vocabulary, 3); err != nil {
		return 0, 0, 0, err
	}
	if !residentFinite(scale) {
		return 0, 0, 0, fmt.Errorf("invalid compact embedding accumulation scale")
	}
	return vocabulary * width, rows * width, 3*rows + 1, nil
}

// ResidentCompactEmbeddingDriver is optional. It does not relax ordinary tape
// binding extents or route any existing scope. The coordinator must admit the
// whole binding/index inventory and every accumulation before invoking it.
// Buffers remain alive through synchronous completion; errors retain owners.
type ResidentCompactEmbeddingDriver interface {
	ValidateCompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index ResidentBuffer, rows, width, vocabulary int, scale float32) error
	CompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index ResidentBuffer, rows, width, vocabulary int, scale float32) error
}
