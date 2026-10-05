package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

func (op *TrainingF32) Vector(dst, a, b *Buffer, n, source, offset int, mode uint32, scale float32) error {
	if err := train.VectorShape(n, source, offset, mode, scale); err != nil {
		return err
	}
	bn := 1
	if mode == 0 {
		bn = n
	}
	if err := dualExtents([]*Buffer{a, b, dst}, []int{source, bn, n}); err != nil {
		return err
	}
	if op == nil {
		return fmt.Errorf("nil PTX training operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.dualModule == nil {
		return fmt.Errorf("dual PTX module unavailable")
	}
	nn, ss, off := uint32(n), uint32(source), uint32(offset)
	if err := LaunchKernel(op.dualModule.Function("training_vector"), uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&a.Ptr), unsafe.Pointer(&b.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&ss), unsafe.Pointer(&off), unsafe.Pointer(&mode), unsafe.Pointer(&scale)); err != nil {
		return err
	}
	return SyncErr()
}
