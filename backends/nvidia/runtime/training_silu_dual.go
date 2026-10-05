package nvidia

import (
	"fmt"
	"unsafe"
)

// SiLUDualBackward returns both input adjoints for the analytic
// reverse-over-forward SiLU operation. Outputs overwrite disjoint buffers.
func (op *TrainingF32) SiLUDualBackward(dx, dt, x, xt, gv, gt *Buffer, n int) error {
	if n < 1 || n > 1<<26 {
		return fmt.Errorf("dual SiLU vector ceiling")
	}
	buffers := []*Buffer{x, xt, gv, gt, dx, dt}
	for _, b := range buffers {
		if !trainingExtent(b, n) {
			return fmt.Errorf("dual SiLU buffer extent")
		}
	}
	for i := 4; i < 6; i++ {
		for j, b := range buffers {
			if i != j && trainingOverlap(buffers[i], n, b, n) {
				return fmt.Errorf("dual SiLU output overlaps")
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
	nn := uint32(n)
	if err := LaunchKernel(op.module.Function("training_silu_dual_backward"), uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&xt.Ptr), unsafe.Pointer(&gv.Ptr), unsafe.Pointer(&gt.Ptr), unsafe.Pointer(&dx.Ptr), unsafe.Pointer(&dt.Ptr), unsafe.Pointer(&nn)); err != nil {
		return err
	}
	return SyncErr()
}
