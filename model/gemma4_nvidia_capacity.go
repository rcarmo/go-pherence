package model

import (
	"errors"
	"fmt"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
)

// ErrGemma4ContextCapacity is an input/resource refusal, never a truncated run.
var ErrGemma4ContextCapacity = errors.New("NVIDIA decision context exceeds capacity")

const gemma4DeviceReserveBytes = 512 << 20

func (g *Gemma4NVIDIA) contextLimit() int {
	if g == nil || g.model == nil {
		return 0
	}
	limit := nvidia.MaxDecisionAttentionTokens
	if g.model.Config.MaxSeqLen > 0 {
		limit = min(limit, g.model.Config.MaxSeqLen)
	}
	return limit
}

// Include both immutable trunk and every private suffix. Check the entire
// allocation before creating the arena, keeping room for projections/readout.
func gemma4ArenaBytes(dims []int, trunk, branches, suffix int) (int, error) {
	if trunk < 1 || trunk > nvidia.MaxDecisionAttentionTokens || branches < 1 || branches > 256 || suffix < 1 || suffix > nvidia.MaxDecisionAttentionTokens {
		return 0, fmt.Errorf("%w: invalid KV dimensions", ErrGemma4ContextCapacity)
	}
	rows := trunk + branches*suffix
	total := 0
	for _, dim := range dims {
		n, ok := checked.MulInt(rows, dim)
		if !ok || dim < 0 {
			return 0, fmt.Errorf("%w: KV size overflow", ErrGemma4ContextCapacity)
		}
		n, ok = checked.MulInt(n, 8)
		if !ok || n > int(^uint(0)>>1)-total {
			return 0, fmt.Errorf("%w: KV size overflow", ErrGemma4ContextCapacity)
		}
		total += n
	}
	return total, nil
}
func checkGemma4ArenaCapacityRows(dims, rows []int, branches, suffix int) error {
	if len(dims) == 0 || len(rows) != len(dims) {
		return fmt.Errorf("%w: invalid KV layout", ErrGemma4ContextCapacity)
	}
	need := 0
	for l, dim := range dims {
		n, err := gemma4ArenaBytes([]int{dim}, rows[l], branches, suffix)
		if err != nil {
			return err
		}
		var ok bool
		need, ok = checked.AddInt(need, n)
		if !ok {
			return fmt.Errorf("%w: KV size overflow", ErrGemma4ContextCapacity)
		}
	}
	free, _ := nvidia.MemInfo()
	if free < uint64(need)+gemma4DeviceReserveBytes {
		return fmt.Errorf("%w: KV needs %d bytes plus %d bytes scratch reserve; device has %d bytes free", ErrGemma4ContextCapacity, need, gemma4DeviceReserveBytes, free)
	}
	return nil
}
