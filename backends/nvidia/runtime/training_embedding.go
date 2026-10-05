package nvidia

import (
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"unsafe"
)

// Embedding requires IDs or an occurrence index admitted by Tape.Validate;
// raw callers are responsible for content admission before dispatch.
func (op *TrainingF32) Embedding(dst, source, ids *Buffer, rows, width, vocabulary int, mode uint32) error {
	a, n, err := train.EmbeddingShape(rows, width, vocabulary, mode)
	if err != nil {
		return err
	}
	indexSize := rows
	if mode == 2 {
		indexSize += vocabulary + 1
	}
	if mode == 3 {
		indexSize = 3*rows + 1
	}
	if err := dualExtents([]*Buffer{source, ids, dst}, []int{a, indexSize, n}); err != nil {
		return err
	}
	if op == nil {
		return fmt.Errorf("nil PTX embedding operator")
	}
	op.mu.Lock()
	defer op.mu.Unlock()
	if op.dualModule == nil || op.poison != nil {
		return fmt.Errorf("closed/terminal PTX embedding operator")
	}
	fn := op.dualModule.Function("training_embedding")
	if fn == 0 {
		return fmt.Errorf("embedding PTX entry unavailable")
	}
	r, w, v := uint32(rows), uint32(width), uint32(vocabulary)
	err = LaunchKernel(fn, uint32((n-1)/256+1), 1, 1, 256, 1, 1, 0, unsafe.Pointer(&source.Ptr), unsafe.Pointer(&ids.Ptr), unsafe.Pointer(&dst.Ptr), unsafe.Pointer(&r), unsafe.Pointer(&w), unsafe.Pointer(&v), unsafe.Pointer(&mode))
	if err == nil {
		err = SyncErr()
	}
	if err != nil {
		op.poison = err
	}
	return err
}
