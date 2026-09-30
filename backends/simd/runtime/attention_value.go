package simd

import "github.com/rcarmo/go-pherence/internal/checked"

// AttentionValueRowTo sums source rows in order using separate float32 multiply
// and add (no FMA/reassociation). dst is overwritten, inputs are immutable and
// may not overlap dst. This keeps the scalar attention reduction's exact bits.
func AttentionValueRowTo(dst, probabilities, values []float32, rows, width int) bool {
	elements, ok := checked.MulInt(rows, width)
	if !ok || rows < 1 || width < 1 || len(dst) < width || len(probabilities) < rows || len(values) < elements {
		return false
	}
	dst = dst[:width]
	probabilities = probabilities[:rows]
	values = values[:elements]
	if !float32SlicesDisjoint(dst, probabilities) || !float32SlicesDisjoint(dst, values) {
		return false
	}
	attentionValueRowTo(dst, probabilities, values, rows, width)
	return true
}
