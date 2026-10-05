package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

func (op *TrainingF32) EOSAdmission(dst, hidden, weight, bias *Buffer, rows, width int) error {
	n, budget, err := train.EOSAdmissionShape(rows, width)
	if err != nil {
		return err
	}
	if err = dualExtents([]*Buffer{hidden, weight, bias, dst}, []int{n, width, 1, 1}); err != nil {
		return err
	}
	if op == nil {
		return fmt.Errorf("nil PTX EOS admission operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.dualModule == nil || op.poison != nil {
		return fmt.Errorf("closed/terminal PTX EOS admission operator")
	}
	fn := op.dualModule.Function("training_eos_admission")
	if fn == 0 {
		return fmt.Errorf("EOS admission PTX entry unavailable")
	}
	r, w := uint32(rows), uint32(width)
	err = LaunchKernel(fn, 1, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&hidden.Ptr), unsafe.Pointer(&weight.Ptr), unsafe.Pointer(&bias.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&w), unsafe.Pointer(&budget))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
