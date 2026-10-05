package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

// GELU applies tanh-approximate GELU or its input VJP. Inputs must keep the
// F32 cubic/intermediate derivative finite (|x|<=1e10 suffices). Raw buffers
// are not host-scanned. Native failures poison the operator; no retry/fallback.
func (op *TrainingF32) GELU(dst, x, seed *Buffer, n int, mode uint32) error {
	if err := train.GELUShape(n, mode); err != nil {
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
	fn := op.dualModule.Function("training_gelu")
	if fn == 0 {
		return fmt.Errorf("GELU PTX entry unavailable")
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

// Row overwrites a gather/scatter, shared-scale multiply or row dot reduction.
func (op *TrainingF32) Row(dst, a, b *Buffer, rows, width, parts, slot int, mode uint32) error {
	an, bn, n, err := train.RowShape(rows, width, parts, slot, mode)
	if err != nil {
		return err
	}
	if err := dualExtents([]*Buffer{a, b, dst}, []int{an, bn, n}); err != nil {
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
	fn := op.dualModule.Function("training_row")
	if fn == 0 {
		return fmt.Errorf("row PTX entry unavailable")
	}
	rr, ww, pp, ss := uint32(rows), uint32(width), uint32(parts), uint32(slot)
	err = LaunchKernel(fn, uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&a.Ptr), unsafe.Pointer(&b.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&rr), unsafe.Pointer(&ww), unsafe.Pointer(&pp), unsafe.Pointer(&ss), unsafe.Pointer(&mode))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
