package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

// AttentionForward overwrites one causal score, probability or output tensor.
// Q/K/V are [S,H,D]; aux is distinct [H,S,S] scores(mode1)/probabilities(mode2)
// or owned scratch(mode0). Whole pass preflight belongs to tape/plan admission.
// Uncertain launches poison this operator; no retry/fallback occurs.
func (op *TrainingF32) AttentionForward(dst, q, k, v, aux *Buffer, rows, heads, dim, contextRows int, mode uint32, scale float32) error {
	if err := train.AttentionForwardShape(rows, heads, dim, contextRows, mode, scale); err != nil {
		return err
	}
	tensor, dense := rows*heads*dim, heads*rows*rows
	elements, count := dense, dense
	if mode == 1 {
		count = heads * rows
	}
	if mode == 2 {
		elements, count = tensor, tensor
	}
	if err := dualExtents([]*Buffer{q, k, v, aux, dst}, []int{tensor, tensor, tensor, dense, elements}); err != nil {
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
	fn := op.dualModule.Function("training_attention_forward")
	if fn == 0 {
		return fmt.Errorf("attention forward PTX entry unavailable")
	}
	rr, hh, dd, cc := uint32(rows), uint32(heads), uint32(dim), uint32(contextRows)
	err := LaunchKernel(fn, uint32((count-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&k.Ptr), unsafe.Pointer(&v.Ptr), unsafe.Pointer(&aux.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&rr), unsafe.Pointer(&hh), unsafe.Pointer(&dd), unsafe.Pointer(&cc), unsafe.Pointer(&mode), unsafe.Pointer(&scale))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
