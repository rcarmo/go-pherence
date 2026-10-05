package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

func (op *TrainingF32) LinearDualForward(dst, x, xt, w, bias *Buffer, m, k, n int, mode uint32) error {
	if err := train.LinearShape(m, k, n); err != nil {
		return err
	}
	if mode > 1 {
		return fmt.Errorf("dual linear forward mode")
	}
	if err := dualExtents([]*Buffer{x, xt, w, bias, dst}, []int{m * k, m * k, n * k, n, m * n}); err != nil {
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
	mm, kk, nn := uint32(m), uint32(k), uint32(n)
	if err := LaunchKernel(op.dualModule.Function("training_linear_dual_forward"), uint32((m*n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&xt.Ptr), unsafe.Pointer(&w.Ptr), unsafe.Pointer(&bias.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&mm), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&mode)); err != nil {
		return err
	}
	return SyncErr()
}
func (op *TrainingF32) SiLUDualForward(dst, x, xt *Buffer, n int, mode uint32) error {
	if n < 1 || n > 1<<26 || mode > 1 {
		return fmt.Errorf("dual SiLU forward count/mode")
	}
	if err := dualExtents([]*Buffer{x, xt, dst}, []int{n, n, n}); err != nil {
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
	nn := uint32(n)
	if err := LaunchKernel(op.dualModule.Function("training_silu_dual_forward"), uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&xt.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&mode)); err != nil {
		return err
	}
	return SyncErr()
}
func (op *TrainingF32) ModDualForward(dst, a, at, b, bt *Buffer, n int, kind, mode uint32) error {
	if err := train.DualModShape(n, kind, mode); err != nil {
		return err
	}
	if mode > 1 {
		return fmt.Errorf("dual modulation forward mode")
	}
	bn := n
	if kind == 0 {
		bn *= 2
	}
	if err := dualExtents([]*Buffer{a, at, b, bt, dst}, []int{n, n, bn, bn, n}); err != nil {
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
	nn := uint32(n)
	if err := LaunchKernel(op.dualModule.Function("training_mod_dual_forward"), uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&a.Ptr), unsafe.Pointer(&at.Ptr), unsafe.Pointer(&b.Ptr), unsafe.Pointer(&bt.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&kind), unsafe.Pointer(&mode)); err != nil {
		return err
	}
	return SyncErr()
}
