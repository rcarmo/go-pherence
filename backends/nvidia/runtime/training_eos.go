package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

func (op *TrainingF32) EOS(dst, x, labels *Buffer, n, active int, mode uint32, weight float32) error {
	out, err := train.EOSShape(n, active, mode, weight)
	if err != nil {
		return err
	}
	if err := dualExtents([]*Buffer{x, labels, dst}, []int{n, n, out}); err != nil {
		return err
	}
	if op == nil {
		return fmt.Errorf("nil PTX EOS operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.dualModule == nil || op.poison != nil {
		return fmt.Errorf("closed/terminal PTX EOS operator")
	}
	fn := op.dualModule.Function("training_eos")
	if fn == 0 {
		return fmt.Errorf("EOS PTX entry unavailable")
	}
	nn, count := uint32(n), uint32(active)
	err = LaunchKernel(fn, uint32((out-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&labels.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&count), unsafe.Pointer(&mode), unsafe.Pointer(&weight))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
