package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

// ReLU overwrites a disjoint forward/VJP vector. Raw inputs must be finite;
// seed is a real owned input even for forward. No implicit runtime/fallback.
func (op *TrainingF32) ReLU(dst, x, seed *Buffer, n int, mode uint32) error {
	if err := train.ReLUShape(n, mode); err != nil {
		return err
	}
	if err := dualExtents([]*Buffer{x, seed, dst}, []int{n, n, n}); err != nil {
		return err
	}
	if op == nil {
		return fmt.Errorf("nil PTX training operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.dualModule == nil || op.poison != nil {
		return fmt.Errorf("closed/terminal PTX training operator")
	}
	fn := op.dualModule.Function("training_relu")
	if fn == 0 {
		return fmt.Errorf("ReLU PTX entry unavailable")
	}
	nn := uint32(n)
	err := LaunchKernel(fn, uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&seed.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&mode))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
