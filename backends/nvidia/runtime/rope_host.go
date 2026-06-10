package nvidia

import (
	"fmt"
	"github.com/rcarmo/go-pherence/internal/checked"
)

// F32RoPE applies rotary position embedding to one or more row-major F32 heads
// through the existing NVIDIA RoPEPartial kernel. cos and sin are separate
// [seqLen, headDim/2] tables, matching Ideogram/Qwen's CPU ropeTable layout.
func F32RoPE(x []float32, cos, sin []float32, pos, nHeads, headDim int) error {
	if pos < 0 || nHeads <= 0 || headDim <= 0 || headDim%2 != 0 {
		return fmt.Errorf("invalid F32 RoPE dims pos=%d heads=%d headDim=%d", pos, nHeads, headDim)
	}
	total, okTotal := checked.MulInt(nHeads, headDim)
	half := headDim / 2
	needTable, okNeed := checked.MulInt(pos+1, half)
	if !okTotal || !okNeed || len(x) < total || len(cos) < needTable || len(sin) < needTable {
		return fmt.Errorf("invalid F32 RoPE buffers x=%d cos=%d sin=%d need x=%d table=%d", len(x), len(cos), len(sin), total, needTable)
	}
	cosSin := make([]float32, needTable*2)
	for i := 0; i < needTable; i++ {
		cosSin[2*i] = cos[i]
		cosSin[2*i+1] = sin[i]
	}
	xBuf := NewDevBufFrom(x[:total])
	csBuf := NewDevBufFrom(cosSin)
	defer xBuf.Free()
	defer csBuf.Free()
	if err := xBuf.ToGPU(); err != nil {
		return err
	}
	if err := csBuf.ToGPU(); err != nil {
		return err
	}
	if !DevRoPEPartial(xBuf, csBuf, pos, nHeads, headDim, half) {
		return fmt.Errorf("F32 RoPE kernel failed")
	}
	copy(x[:total], xBuf.Data()[:total])
	return nil
}
