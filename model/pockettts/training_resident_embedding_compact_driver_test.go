//go:build linux

package pockettts

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
)

// CPU/mock capability used only by explicit compact tests. Existing scopes
// retain their old driver identity/behaviour except the unused extra methods.
func (d *hostResidentGraphDriver) ValidateCompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index train.ResidentBuffer, rows, width, vocabulary int, scale float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	dense, slots, indices, err := train.CompactEmbeddingAccumulateShape(rows, width, vocabulary, scale)
	if err != nil {
		return err
	}
	buffers := []train.ResidentBuffer{out, previous, compact, index}
	for i, b := range buffers {
		x, ok := b.(*hostOptimizerBuffer)
		if !ok || len(x.values) != []int{dense, dense, slots, indices}[i] {
			return fmt.Errorf("mock compact accumulation extent")
		}
		if i > 0 && f32SlicesOverlap(buffers[0].(*hostOptimizerBuffer).values, x.values) {
			return fmt.Errorf("mock writable overlap")
		}
	}
	d.accumulations++
	return nil
}
func (d *hostResidentGraphDriver) CompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index train.ResidentBuffer, rows, width, vocabulary int, scale float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	o, p, c, idx := out.(*hostOptimizerBuffer).values, previous.(*hostOptimizerBuffer).values, compact.(*hostOptimizerBuffer).values, index.(*hostOptimizerBuffer).values
	for i := range o {
		token := i / width
		lo, hi := 0, rows
		for lo < hi {
			mid := lo + (hi-lo)/2
			id := idx[mid]
			if id >= 0 && id < float32(token) {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		var contribution float32
		if lo < rows && idx[lo] == float32(token) {
			contribution = c[lo*width+i%width]
		}
		o[i] = p[i] + scale*contribution
		if math.IsNaN(float64(o[i])) {
			return fmt.Errorf("mock compact nonfinite")
		}
	}
	if d.afterAccumulate != nil {
		d.afterAccumulate(o)
	}
	return nil
}
