package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

func (op *TrainingF32) NormDualForward(dst, x, xt, w, bias, stats *Buffer, m, n int, kind, mode uint32, epsilon float32) error {
	if err := train.DualNormForwardShape(m, n, kind, mode, epsilon); err != nil {
		return err
	}
	elements, count := m*n, m*n
	if mode == 0 {
		elements = m * 5
		count = m
	}
	if err := dualExtents([]*Buffer{x, xt, w, bias, stats, dst}, []int{m * n, m * n, n, n, m * 5, elements}); err != nil {
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
	mm, nn := uint32(m), uint32(n)
	if err := LaunchKernel(op.dualModule.Function("training_norm_dual_forward"), uint32((count-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&xt.Ptr), unsafe.Pointer(&w.Ptr), unsafe.Pointer(&bias.Ptr), unsafe.Pointer(&stats.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&mm), unsafe.Pointer(&nn), unsafe.Pointer(&kind), unsafe.Pointer(&mode), unsafe.Pointer(&epsilon)); err != nil {
		return err
	}
	return SyncErr()
}
