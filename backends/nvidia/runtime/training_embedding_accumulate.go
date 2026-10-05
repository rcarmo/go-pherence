package nvidia

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

func (op *TrainingF32) CompactEmbeddingAccumulate(out, previous, compact, index *Buffer, rows, width, vocabulary int, scale float32) error {
	dense, slots, indices, err := train.CompactEmbeddingAccumulateShape(rows, width, vocabulary, scale)
	if err != nil {
		return err
	}
	if err := dualExtents([]*Buffer{previous, compact, index, out}, []int{dense, slots, indices, dense}); err != nil {
		return err
	}
	if op == nil {
		return fmt.Errorf("nil PTX compact accumulation operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.dualModule == nil || op.poison != nil {
		return fmt.Errorf("closed/terminal PTX compact accumulation operator")
	}
	fn := op.dualModule.Function("training_embedding_accumulate")
	if fn == 0 {
		return fmt.Errorf("compact accumulation PTX entry unavailable")
	}
	r, w, v := uint32(rows), uint32(width), uint32(vocabulary)
	err = LaunchKernel(fn, uint32((dense-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&previous.Ptr), unsafe.Pointer(&compact.Ptr), unsafe.Pointer(&index.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&w), unsafe.Pointer(&v), unsafe.Pointer(&scale))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}

var _ train.ResidentCompactEmbeddingDriver = (*PTXResidentDriver)(nil)

func (d *PTXResidentDriver) ValidateCompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index train.ResidentBuffer, rows, width, vocabulary int, scale float32) error {
	if err := d.admit(ctx); err != nil {
		return err
	}
	dense, slots, indices, err := train.CompactEmbeddingAccumulateShape(rows, width, vocabulary, scale)
	if err != nil {
		return err
	}
	buffers, err := ptxResidentBuffers(previous, compact, index, out)
	if err != nil {
		return err
	}
	if err := dualExtents(buffers, []int{dense, slots, indices, dense}); err != nil {
		return err
	}
	d.Operator.mu.Lock()
	defer d.Operator.mu.Unlock()
	if d.Operator.poison != nil || d.Operator.dualModule == nil || d.Operator.dualModule.Function("training_embedding_accumulate") == 0 {
		return fmt.Errorf("compact accumulation PTX unavailable")
	}
	return nil
}
func (d *PTXResidentDriver) CompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index train.ResidentBuffer, rows, width, vocabulary int, scale float32) error {
	if err := d.ValidateCompactEmbeddingAccumulate(ctx, out, previous, compact, index, rows, width, vocabulary, scale); err != nil {
		return err
	}
	buffers, err := ptxResidentBuffers(out, previous, compact, index)
	if err != nil {
		return err
	}
	return d.Operator.CompactEmbeddingAccumulate(buffers[0], buffers[1], buffers[2], buffers[3], rows, width, vocabulary, scale)
}
