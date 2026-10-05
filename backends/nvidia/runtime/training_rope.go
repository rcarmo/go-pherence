package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

// RoPE applies adjacent-pair forward or transpose rotation to owned F32 storage.
// Tables are caller-computed [rows,dim/2,2] cosine,sine values; no device trig
// approximation or CPU fallback is implicit. Any attempted launch/sync failure
// poisons this operator; caller must retain buffers and recover coherently.
func (op *TrainingF32) RoPE(dst, x, table *Buffer, rows, heads, dim int, mode uint32) error {
	if err := train.RoPEShape(rows, heads, dim, mode); err != nil {
		return err
	}
	count := rows * heads * dim
	if err := dualExtents([]*Buffer{x, table, dst}, []int{count, rows * dim, count}); err != nil {
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
	fn := op.dualModule.Function("training_rope")
	if fn == 0 {
		return fmt.Errorf("RoPE PTX entry unavailable")
	}
	rr, hh, dd := uint32(rows), uint32(heads), uint32(dim)
	err := LaunchKernel(fn, uint32((count/2-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&table.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&rr), unsafe.Pointer(&hh), unsafe.Pointer(&dd), unsafe.Pointer(&mode))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
