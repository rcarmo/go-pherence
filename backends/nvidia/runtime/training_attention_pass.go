package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// AttentionBackwardPass validates all scratch/output/input ranges before any
// launch, then evaluates five ordered stages. All writes are disjoint. This
// wrapper synchronises each stage; resident launch fusion is a separate future
// optimisation, not a throughput claim. dQ/dK need inverse RoPE afterwards.
func (op *TrainingF32) AttentionBackwardPass(dq, dk, dv, dp, ds, q, k, v, prob, dy *Buffer, rows, heads, dim, contextRows int, scale float32) error {
	if err := train.AttentionShape(rows, heads, dim, contextRows, 0, scale); err != nil {
		return err
	}
	tensor, dense := rows*heads*dim, heads*rows*rows
	buffers := []*Buffer{dq, dk, dv, dp, ds, q, k, v, prob, dy}
	sizes := []int{tensor, tensor, tensor, dense, dense, tensor, tensor, tensor, dense, tensor}
	for i, b := range buffers {
		if !trainingExtent(b, sizes[i]) {
			return fmt.Errorf("PTX attention pass extent")
		}
	}
	for i := 0; i < 5; i++ {
		for j := range buffers {
			if i != j && trainingOverlap(buffers[i], sizes[i], buffers[j], sizes[j]) {
				return fmt.Errorf("PTX attention pass write overlaps")
			}
		}
	}
	if op == nil {
		return fmt.Errorf("nil PTX training operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.module == nil {
		return fmt.Errorf("closed PTX training operator")
	}
	for _, stage := range []struct {
		dst, aux *Buffer
		mode     uint32
		count    int
	}{{dp, ds, 0, dense}, {ds, dp, 1, rows * heads}, {dq, ds, 2, tensor}, {dk, ds, 3, tensor}, {dv, ds, 4, tensor}} {
		if err := op.attentionBackwardLocked(stage.dst, q, k, v, prob, dy, stage.aux, rows, heads, dim, contextRows, stage.mode, scale, stage.count); err != nil {
			return err
		}
	}
	return nil
}
