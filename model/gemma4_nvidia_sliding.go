package model

import (
	"fmt"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// Keep only the preceding window plus one prefill chunk. Absolute RoPE positions
// are unchanged; attention receives offsets relative to the retained KV base.
func (a *gemma4NVIDIAKVArena) prepareTrunkWrite(layer, pos, rows int) error {
	base, capacity, window := a.trunkBase[layer], a.trunkRows[layer], a.windows[layer]
	if pos < base {
		return fmt.Errorf("NVIDIA sliding KV rewind requires prefix rebuild")
	}
	if pos+rows-base <= capacity {
		return nil
	}
	if window <= 0 || rows > Gemma4PackedRows {
		return fmt.Errorf("NVIDIA KV write exceeds allocation")
	}
	next := max(base, pos-window+1)
	keep := pos - next
	dim := a.kvDim[layer]
	if keep+rows > capacity {
		return fmt.Errorf("NVIDIA sliding KV capacity exceeded")
	}
	if keep > 0 {
		// CUDA DtoD copies cannot overlap. Stage the retained tail explicitly.
		scratch, err := nvidia.Malloc(keep * dim)
		if err != nil {
			return err
		}
		defer scratch.Free()
		defer nvidia.SyncAll()
		for _, buf := range []*nvidia.Buffer{a.trunkK[layer], a.trunkV[layer]} {
			if err = nvidia.CopyDtoD(scratch.Ptr, buf.Ptr+nvidia.CUdeviceptr((next-base)*dim*4), uint64(keep*dim*4)); err != nil {
				return err
			}
			if err = nvidia.CopyDtoD(buf.Ptr, scratch.Ptr, uint64(keep*dim*4)); err != nil {
				return err
			}
		}
	}
	a.trunkBase[layer] = next
	return nil
}

// A compacted arena no longer contains the old schema-prefix KV. It cannot be
// reused for a different context even when the cached token IDs match.
func (c *Gemma4NVIDIAContext) CanReusePrefix() bool {
	if c == nil || c.closed || c.arena == nil {
		return false
	}
	for _, base := range c.arena.trunkBase {
		if base != 0 {
			return false
		}
	}
	return true
}
