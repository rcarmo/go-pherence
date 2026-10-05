package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

func (op *TrainingF32) LSDLoss(dst, prediction, desired, logvar *Buffer, channels int, mode uint32) error {
	n, err := train.LSDLossShape(channels, mode)
	if err != nil {
		return err
	}
	if err := dualExtents([]*Buffer{prediction, desired, logvar, dst}, []int{channels, channels, 1, n}); err != nil {
		return err
	}
	if op == nil {
		return fmt.Errorf("nil PTX LSD loss operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.dualModule == nil || op.poison != nil {
		return fmt.Errorf("closed/terminal PTX LSD loss operator")
	}
	fn := op.dualModule.Function("training_lsd_loss")
	if fn == 0 {
		return fmt.Errorf("LSD loss PTX entry unavailable")
	}
	c := uint32(channels)
	err = LaunchKernel(fn, uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&prediction.Ptr), unsafe.Pointer(&desired.Ptr), unsafe.Pointer(&logvar.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&c), unsafe.Pointer(&mode))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
