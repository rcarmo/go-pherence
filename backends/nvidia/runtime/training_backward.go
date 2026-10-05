package nvidia

import (
	"fmt"
	"unsafe"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func (op *TrainingF32) LayerNormBackward(dst, xhat, inv, weight, dy *Buffer, rows, width int, mode uint32) error {
	if err := train.LayerNormShape(rows, width, mode); err != nil {
		return err
	}
	count, elements := width, width
	if mode == 0 {
		count = rows
		elements = rows * width
	}
	if !trainingExtent(dst, elements) || !trainingExtent(xhat, rows*width) || !trainingExtent(inv, rows) || !trainingExtent(weight, width) || !trainingExtent(dy, rows*width) {
		return fmt.Errorf("PTX layernorm backward extent")
	}
	for _, in := range []struct {
		b *Buffer
		n int
	}{{xhat, rows * width}, {inv, rows}, {weight, width}, {dy, rows * width}} {
		if trainingOverlap(dst, elements, in.b, in.n) {
			return fmt.Errorf("PTX layernorm output overlaps")
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
	m, n := uint32(rows), uint32(width)
	if err := LaunchKernel(op.module.Function("training_layernorm_backward"), uint32((count-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&xhat.Ptr), unsafe.Pointer(&inv.Ptr), unsafe.Pointer(&weight.Ptr), unsafe.Pointer(&dy.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&m), unsafe.Pointer(&n), unsafe.Pointer(&mode)); err != nil {
		return err
	}
	return SyncErr()
}

// AttentionBackward is one overwrite stage of causal attention differentiation.
// Buffer layouts and mode semantics are defined by training.AttentionShape.
// Aux remains a valid dense tensor when unused. No implicit masking/padding
// shortcut, RoPE application, atomic addition or CPU fallback occurs.
func (op *TrainingF32) AttentionBackward(dst, q, k, v, prob, dy, aux *Buffer, rows, heads, dim, contextRows int, mode uint32, scale float32) error {
	if err := train.AttentionShape(rows, heads, dim, contextRows, mode, scale); err != nil {
		return err
	}
	tensor, dense := rows*heads*dim, heads*rows*rows
	elements, count := tensor, tensor
	if mode < 2 {
		elements = dense
		count = dense
		if mode == 1 {
			count = rows * heads
		}
	}
	if !trainingExtent(dst, elements) {
		return fmt.Errorf("PTX attention destination extent")
	}
	for _, in := range []struct {
		b *Buffer
		n int
	}{{q, tensor}, {k, tensor}, {v, tensor}, {prob, dense}, {dy, tensor}, {aux, dense}} {
		if !trainingExtent(in.b, in.n) {
			return fmt.Errorf("PTX attention input extent")
		}
		if trainingOverlap(dst, elements, in.b, in.n) {
			return fmt.Errorf("PTX attention output overlaps")
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
	return op.attentionBackwardLocked(dst, q, k, v, prob, dy, aux, rows, heads, dim, contextRows, mode, scale, count)
}
func (op *TrainingF32) attentionBackwardLocked(dst, q, k, v, prob, dy, aux *Buffer, rows, heads, dim, contextRows int, mode uint32, scale float32, count int) error {
	s, h, d, c := uint32(rows), uint32(heads), uint32(dim), uint32(contextRows)
	if err := LaunchKernel(op.module.Function("training_attention_backward"), uint32((count-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&k.Ptr), unsafe.Pointer(&v.Ptr), unsafe.Pointer(&prob.Ptr), unsafe.Pointer(&dy.Ptr), unsafe.Pointer(&aux.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&s), unsafe.Pointer(&h), unsafe.Pointer(&d), unsafe.Pointer(&c), unsafe.Pointer(&mode), unsafe.Pointer(&scale)); err != nil {
		return err
	}
	return SyncErr()
}
