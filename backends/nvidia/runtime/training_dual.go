package nvidia

import (
	"fmt"
	"unsafe"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func (op *TrainingF32) LinearDualBackward(dst, x, xt, w, gv, gt *Buffer, m, k, n int, mode uint32) error {
	if err := train.LinearShape(m, k, n); err != nil {
		return err
	}
	if mode > 3 {
		return fmt.Errorf("dual linear mode")
	}
	count := n
	if mode < 2 {
		count = m * k
	} else if mode == 2 {
		count = n * k
	}
	buffers := []*Buffer{x, xt, w, gv, gt, dst}
	sizes := []int{m * k, m * k, n * k, m * n, m * n, count}
	if err := dualExtents(buffers, sizes); err != nil {
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
	if err := LaunchKernel(op.dualModule.Function("training_linear_dual_backward"), uint32((count-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&xt.Ptr), unsafe.Pointer(&w.Ptr), unsafe.Pointer(&gv.Ptr), unsafe.Pointer(&gt.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&mm), unsafe.Pointer(&kk), unsafe.Pointer(&nn), unsafe.Pointer(&mode)); err != nil {
		return err
	}
	return SyncErr()
}
func (op *TrainingF32) NormDualBackward(dst, x, xt, w, gv, gt, stats *Buffer, m, n int, kind, mode uint32) error {
	if err := train.DualNormShape(m, n, kind, mode); err != nil {
		return err
	}
	count, elements := n, n
	if mode < 2 {
		count = m
		elements = m * n
	}
	buffers := []*Buffer{x, xt, w, gv, gt, stats, dst}
	sizes := []int{m * n, m * n, n, m * n, m * n, m * 5, elements}
	if err := dualExtents(buffers, sizes); err != nil {
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
	if err := LaunchKernel(op.dualModule.Function("training_norm_dual_backward"), uint32((count-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&xt.Ptr), unsafe.Pointer(&w.Ptr), unsafe.Pointer(&gv.Ptr), unsafe.Pointer(&gt.Ptr), unsafe.Pointer(&stats.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&mm), unsafe.Pointer(&nn), unsafe.Pointer(&kind), unsafe.Pointer(&mode)); err != nil {
		return err
	}
	return SyncErr()
}
func (op *TrainingF32) ModDualBackward(dst, a, at, b, bt, gv, gt *Buffer, n int, kind, mode uint32) error {
	if err := train.DualModShape(n, kind, mode); err != nil {
		return err
	}
	bn, count := n, n
	if kind == 0 {
		bn = 2 * n
		if mode >= 2 {
			count = 2 * n
		}
	}
	if err := dualExtents([]*Buffer{a, at, b, bt, gv, gt, dst}, []int{n, n, bn, bn, n, n, count}); err != nil {
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
	if err := LaunchKernel(op.dualModule.Function("training_mod_dual_backward"), uint32((count-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&a.Ptr), unsafe.Pointer(&at.Ptr), unsafe.Pointer(&b.Ptr), unsafe.Pointer(&bt.Ptr), unsafe.Pointer(&gv.Ptr), unsafe.Pointer(&gt.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&nn), unsafe.Pointer(&kind), unsafe.Pointer(&mode)); err != nil {
		return err
	}
	return SyncErr()
}
func dualExtents(buffers []*Buffer, sizes []int) error {
	for i, b := range buffers {
		if !trainingExtent(b, sizes[i]) {
			return fmt.Errorf("dual training tensor extent")
		}
	}
	last := len(buffers) - 1
	for i, b := range buffers[:last] {
		if trainingOverlap(buffers[last], sizes[last], b, sizes[i]) {
			return fmt.Errorf("dual training output overlaps")
		}
	}
	return nil
}
