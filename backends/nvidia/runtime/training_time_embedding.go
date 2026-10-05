package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

func (op *TrainingF32) TimeEmbedding(dst, frequency, time, gv, gt *Buffer, half int, mode uint32) error {
	n, err := train.TimeEmbeddingShape(half, mode)
	if err != nil {
		return err
	}
	if err := dualExtents([]*Buffer{frequency, time, gv, gt, dst}, []int{half, 2, 2 * half, 2 * half, n}); err != nil {
		return err
	}
	if op == nil {
		return fmt.Errorf("nil PTX time embedding operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.dualModule == nil || op.poison != nil {
		return fmt.Errorf("closed/terminal PTX time embedding operator")
	}
	fn := op.dualModule.Function("training_time_embedding")
	if fn == 0 {
		return fmt.Errorf("time embedding PTX entry unavailable")
	}
	h := uint32(half)
	err = LaunchKernel(fn, uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&frequency.Ptr), unsafe.Pointer(&time.Ptr), unsafe.Pointer(&gv.Ptr), unsafe.Pointer(&gt.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&h), unsafe.Pointer(&mode))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
