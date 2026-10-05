package pockettts

import (
	"context"
	"errors"
	"fmt"
	"math"

	train "github.com/rcarmo/go-pherence/backends/training"
	"github.com/rcarmo/go-pherence/backends/vulkan"
)

// VulkanTrainingAttention is a transfer-inclusive hybrid of CPU model forward
// and Vulkan attention backward. Caller explicitly initialises Vulkan and owns
// Operator. No implicit native setup or CPU fallback. Allocation is bounded per
// call; failed cleanup retains arena/plan ownership until explicit disposition.
// It is single-owner and not a complete resident GPU model.
type VulkanTrainingAttention struct {
	Context        context.Context
	Operator       *vulkan.VkTrainingF32
	MaxBytes       int64
	terminal       error
	retainedArenas []*vulkan.VkTensorArena
	retainedPlans  []*vulkan.VkF32Plan
}

func (b *VulkanTrainingAttention) Close() error {
	if b == nil {
		return nil
	}
	if b.terminal != nil {
		return fmt.Errorf("terminal Vulkan attention resources retained: %w", b.terminal)
	}
	return b.Operator.Close()
}
func (b *VulkanTrainingAttention) TrainingAttentionBackendName() string {
	return "vulkan-attention-hybrid-v1"
}

func (b *VulkanTrainingAttention) AttentionBackward(q, k, v, prob, dy []float32, rows, heads, dim, contextRows int) (dq, dk, dv []float32, err error) {
	if b == nil || b.Context == nil || b.Operator == nil || b.MaxBytes <= 0 || b.terminal != nil {
		return nil, nil, nil, fmt.Errorf("invalid/terminal Vulkan attention backend")
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
		return nil, nil, nil, fmt.Errorf("Vulkan attention host shape mismatch")
	}
	for _, input := range [][]float32{q, k, v, prob, dy} {
		if !finiteF32(input) {
			return nil, nil, nil, fmt.Errorf("nonfinite Vulkan attention input")
		}
	}
	shapes := [][]int{{rows, heads, dim}, {rows, heads, dim}, {rows, heads, dim}, {heads, rows, rows}, {heads, rows, rows}, {rows, heads, dim}, {rows, heads, dim}, {rows, heads, dim}, {heads, rows, rows}, {rows, heads, dim}}
	// Each tensor has its own arena and offset0, avoiding guessed descriptor
	// alignment and one arena crossing maxStorageBufferRange. MaxBytes charges
	// logical tensor bytes; Vulkan's allocator separately enforces actual
	// memory-requirement padding/device budget. It is not a peak RSS guarantee.
	total := int64(7*tensor+3*dense) * 4
	if total > b.MaxBytes {
		return nil, nil, nil, fmt.Errorf("Vulkan attention allocation ceiling")
	}
	if !b.Operator.AttentionReady() {
		return nil, nil, nil, fmt.Errorf("Vulkan attention operator not ready; explicit initialisation required")
	}
	var arenas []*vulkan.VkTensorArena
	var tensors []*vulkan.VkTensorF32
	var plan *vulkan.VkF32Plan
	defer func() {
		if b.terminal != nil {
			b.retainedArenas = append(b.retainedArenas, arenas...)
			if plan != nil {
				b.retainedPlans = append(b.retainedPlans, plan)
			}
			dq, dk, dv = nil, nil, nil
			return
		}
		if plan != nil {
			if closeErr := plan.Close(); closeErr != nil {
				b.terminal = closeErr
				b.retainedPlans = append(b.retainedPlans, plan)
				b.retainedArenas = append(b.retainedArenas, arenas...)
				err = errors.Join(err, closeErr)
				dq, dk, dv = nil, nil, nil
				return
			}
		}
		for _, arena := range arenas {
			if closeErr := arena.Close(); closeErr != nil {
				b.retainedArenas = append(b.retainedArenas, arena)
				b.terminal = errors.Join(b.terminal, closeErr)
				err = errors.Join(err, closeErr)
			}
		}
		if err != nil {
			dq, dk, dv = nil, nil, nil
		}
	}()
	for _, shape := range shapes {
		n := shape[0] * shape[1] * shape[2]
		arena, allocErr := vulkan.NewVkTensorArena(b.Context, n*4)
		if allocErr != nil {
			return nil, nil, nil, allocErr
		}
		arenas = append(arenas, arena)
		tensor, allocErr := arena.AllocF32(b.Context, shape...)
		if allocErr != nil {
			return nil, nil, nil, allocErr
		}
		tensors = append(tensors, tensor)
	}
	for i, input := range [][]float32{q, k, v, prob, dy} {
		if err = tensors[5+i].Upload(b.Context, input); err != nil {
			return nil, nil, nil, err
		}
	}
	plan, err = b.Operator.AttentionBackwardPlan(b.Context, tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6], tensors[7], tensors[8], tensors[9], contextRows, scale)
	if err != nil {
		return nil, nil, nil, err
	}
	if err = plan.Run(b.Context); err != nil {
		b.terminal = err
		return nil, nil, nil, err
	}
	dq, dk, dv = make([]float32, tensor), make([]float32, tensor), make([]float32, tensor)
	for i, output := range [][]float32{dq, dk, dv} {
		if err = tensors[i].Download(b.Context, output); err != nil {
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
