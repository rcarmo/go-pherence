package training

import (
	"context"
	"fmt"
)

// ResidentCompactEmbeddingBinding accompanies an ordinary parameter/gradient
// ID pair only through the explicit compact coordinator API. Index is an
// admitted immutable constant and declared output, retained for accumulation.
// Gradient must be produced directly by the matching mode3 embedding command.
type ResidentCompactEmbeddingBinding struct{ Index, Rows, Width, Vocabulary int }

func ValidateResidentCompactTapeBindings(t Tape, bindings map[string]ResidentTapeBinding, compact map[string]ResidentCompactEmbeddingBinding, maxBytes int64) error {
	if compact == nil {
		return fmt.Errorf("compact resident bindings required")
	}
	return validateResidentTapeBindings(t, bindings, compact, maxBytes)
}

func validateResidentCompactBindings(t Tape, bindings map[string]ResidentTapeBinding, compact map[string]ResidentCompactEmbeddingBinding) error {
	if len(compact) == 0 {
		return fmt.Errorf("compact resident bindings required")
	}
	declared := map[int]bool{}
	for _, id := range t.Outputs {
		declared[id] = true
	}
	for name, c := range compact {
		b, ok := bindings[name]
		if !ok || c.Index < 0 || c.Index >= len(t.Tensors) || !declared[c.Index] || c.Index == b.Parameter || c.Index == b.Gradient {
			return fmt.Errorf("compact resident binding name/index")
		}
		for _, other := range bindings {
			if c.Index == other.Parameter {
				return fmt.Errorf("compact resident index rebound as parameter")
			}
		}
		dense, slots, indices, err := CompactEmbeddingAccumulateShape(c.Rows, c.Width, c.Vocabulary, 1)
		if err != nil {
			return err
		}
		p, _ := t.Tensors[b.Parameter].Elements()
		g, _ := t.Tensors[b.Gradient].Elements()
		i, _ := t.Tensors[c.Index].Elements()
		if p != dense || g != slots || i != indices || t.Tensors[c.Index].Values == nil {
			return fmt.Errorf("compact resident binding extent/constant")
		}
		if err := ValidateCompactEmbeddingIndex(t.Tensors[c.Index].Values, c.Rows, c.Vocabulary); err != nil {
			return err
		}
		found := false
		for _, op := range t.Commands {
			if op.Output == b.Gradient {
				if op.Op != "embedding" || op.Mode != 3 || op.Rows != c.Rows || op.Width != c.Width || op.Parts != c.Vocabulary || len(op.Inputs) != 2 || op.Inputs[1] != c.Index {
					return fmt.Errorf("compact resident gradient provenance")
				}
				found = true
			}
		}
		if !found {
			return fmt.Errorf("compact resident gradient producer")
		}
	}
	return nil
}

// RunCompactZeroPrunedTape explicitly enables mixed dense and compact binding
// accumulation. It shares legacy borrowing/terminal behaviour but retains both
// compact gradient and index roots through pruning, without dense expansion.
func (s *ResidentState) RunCompactZeroPrunedTape(ctx context.Context, t Tape, bindings map[string]ResidentTapeBinding, compact map[string]ResidentCompactEmbeddingBinding, outputs []int, accumulate bool, scale float32, maxBytes int64) ([][]float32, error) {
	if compact == nil {
		return nil, fmt.Errorf("compact resident bindings required")
	}
	return s.runTapeBindings(ctx, t, bindings, compact, outputs, accumulate, scale, maxBytes, false, true, true, true)
}
