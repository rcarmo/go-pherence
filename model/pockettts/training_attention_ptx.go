package pockettts

import (
	"context"
	"errors"
	"fmt"
	"math"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	train "github.com/rcarmo/go-pherence/backends/training"
)

// PTXTrainingAttention is an explicit transfer-inclusive attention-backward
// hybrid. It does not retain a model or imply full GPU training. Caller supplies
// a live TrainingF32 operator and a fixed allocation ceiling. No Init occurs here.
// Single-owner, no concurrent calls/Close. Failed frees retain their buffer
// ownership in this object; a terminal backend requires explicit recovery.
type PTXTrainingAttention struct {
	Context  context.Context
	Operator *nvidia.TrainingF32
	MaxBytes int64
	terminal error
	retained []*nvidia.Buffer
}

// Close refuses native cleanup after terminal uncertainty and retains owners.
func (b *PTXTrainingAttention) Close() error {
	if b == nil {
		return nil
	}
	if b.terminal != nil {
		return fmt.Errorf("terminal PTX attention resources retained: %w", b.terminal)
	}
	return b.Operator.Close()
}
func (b *PTXTrainingAttention) TrainingAttentionBackendName() string {
	return "ptx-attention-hybrid-v1"
}

func (b *PTXTrainingAttention) AttentionBackward(q, k, v, prob, dy []float32, rows, heads, dim, contextRows int) (dq, dk, dv []float32, err error) {
	if b == nil || b.Context == nil || b.Operator == nil || b.MaxBytes <= 0 || b.terminal != nil {
		return nil, nil, nil, fmt.Errorf("invalid/terminal PTX attention backend")
	}
	if err = b.Context.Err(); err != nil {
		return nil, nil, nil, err
	}
	scale := float32(1 / math.Sqrt(float64(dim)))
	if err = train.AttentionShape(rows, heads, dim, min(contextRows, rows), 0, scale); err != nil {
		return nil, nil, nil, err
	}
	contextRows = min(contextRows, rows)
	tensor, dense := rows*heads*dim, heads*rows*rows
	if len(q) != tensor || len(k) != tensor || len(v) != tensor || len(dy) != tensor || len(prob) != dense {
		return nil, nil, nil, fmt.Errorf("PTX attention host shape mismatch")
	}
	for _, input := range [][]float32{q, k, v, prob, dy} {
		if !finiteF32(input) {
			return nil, nil, nil, fmt.Errorf("nonfinite PTX attention input")
		}
	}
	sizes := []int{tensor, tensor, tensor, dense, dense, tensor, tensor, tensor, dense, tensor}
	total := int64(0)
	for _, n := range sizes {
		total += int64(n) * 4
	}
	if total > b.MaxBytes {
		return nil, nil, nil, fmt.Errorf("PTX attention allocation ceiling")
	}
	if !b.Operator.AttentionReady() {
		return nil, nil, nil, fmt.Errorf("PTX attention operator not ready; explicit initialisation required")
	}
	var buffers []*nvidia.Buffer
	defer func() {
		// A device/launch/transfer failure is not proof that queued work finished.
		// Retain every referenced buffer instead of freeing uncertain storage.
		if b.terminal != nil {
			b.retained = append(b.retained, buffers...)
			dq, dk, dv = nil, nil, nil
			return
		}
		for _, buffer := range buffers {
			if freeErr := buffer.FreeChecked(); freeErr != nil {
				b.retained = append(b.retained, buffer)
				b.terminal = errors.Join(b.terminal, freeErr)
				err = errors.Join(err, freeErr)
			}
		}
		if err != nil {
			dq, dk, dv = nil, nil, nil
		}
	}()
	for _, n := range sizes {
		buffer, allocErr := nvidia.Malloc(n)
		if allocErr != nil {
			b.terminal = allocErr
			return nil, nil, nil, allocErr
		}
		buffers = append(buffers, buffer)
	}
	for i, input := range [][]float32{q, k, v, prob, dy} {
		if err = buffers[5+i].Upload(input); err != nil {
			b.terminal = err
			return nil, nil, nil, err
		}
	}
	if err = b.Operator.AttentionBackwardPass(buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], buffers[6], buffers[7], buffers[8], buffers[9], rows, heads, dim, contextRows, scale); err != nil {
		b.terminal = err
		return nil, nil, nil, err
	}
	dq, dk, dv = make([]float32, tensor), make([]float32, tensor), make([]float32, tensor)
	for i, output := range [][]float32{dq, dk, dv} {
		if err = buffers[i].Download(output); err != nil {
			b.terminal = err
			return nil, nil, nil, err
		}
	}
	if err = validateAttentionDerivatives(dq, dk, dv, tensor); err != nil {
		b.terminal = err
		return nil, nil, nil, err
	}
	return dq, dk, dv, nil
}
