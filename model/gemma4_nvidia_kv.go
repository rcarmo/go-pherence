package model

import (
	"fmt"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// gemma4NVIDIAKVArena stores one immutable trunk and only branch-local suffix
// rows. This keeps memory O(trunk + branches*suffix), not O(branches*trunk).
type gemma4NVIDIAKVArena struct {
	trunkK, trunkV                []*nvidia.Buffer
	suffixK, suffixV              []*nvidia.Buffer
	kvDim                         []int
	trunkBase, trunkRows, windows []int // physical sliding-cache metadata
	branches                      int
	trunkLen                      int // logical capacity; trunkRows bounds physical storage
	trunkUsed                     int // logical prefix+context rows
	suffixCap                     int
}

func newGemma4NVIDIAKVArena(m *LlamaModel, trunk MTPPromptContext, branches, maxDepth int) (*gemma4NVIDIAKVArena, error) {
	if trunk.SeqLen <= 0 {
		return nil, fmt.Errorf("invalid NVIDIA KV trunk length")
	}
	a, err := allocGemma4NVIDIAKVArena(m, trunk.SeqLen, branches, maxDepth)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			a.free()
		}
	}()
	for l, dim := range a.kvDim {
		if dim == 0 {
			continue
		}
		if len(trunk.KVCacheK) <= l || len(trunk.KVCacheV) <= l || len(trunk.KVCacheK[l]) != trunk.SeqLen*dim || len(trunk.KVCacheV[l]) != trunk.SeqLen*dim {
			return nil, fmt.Errorf("invalid NVIDIA KV trunk layer %d", l)
		}
		a.trunkBase[l] = trunk.SeqLen - a.trunkRows[l]
		if err = a.trunkK[l].Upload(trunk.KVCacheK[l][(trunk.SeqLen-a.trunkRows[l])*dim:]); err != nil {
			return nil, err
		}
		if err = a.trunkV[l].Upload(trunk.KVCacheV[l][(trunk.SeqLen-a.trunkRows[l])*dim:]); err != nil {
			return nil, err
		}
	}
	ok = true
	return a, nil
}

func allocGemma4NVIDIAKVArena(m *LlamaModel, trunkLen, branches, maxDepth int) (*gemma4NVIDIAKVArena, error) {
	if m == nil || branches <= 0 || trunkLen <= 0 || maxDepth <= 0 {
		return nil, fmt.Errorf("invalid NVIDIA KV arena shape")
	}
	dims := make([]int, m.Config.NumLayers)
	for l := range dims {
		var err error
		dims[l], err = m.LayerKVDim(l)
		if err != nil {
			return nil, err
		}
	}
	windows := make([]int, len(dims))
	for l := range windows {
		if len(m.Config.LayerTypes) > l && m.Config.LayerTypes[l] == "sliding_attention" {
			windows[l] = m.Config.SlidingWindow
		}
	}
	return allocGemma4NVIDIAKVArenaWindowed(dims, trunkLen, branches, maxDepth, windows)
}
func allocGemma4NVIDIAKVArenaWindowed(dims []int, trunkLen, branches, maxDepth int, windows []int) (*gemma4NVIDIAKVArena, error) {
	if len(dims) == 0 || len(windows) != len(dims) || trunkLen < 1 || trunkLen > nvidia.MaxDecisionAttentionTokens {
		return nil, fmt.Errorf("%w: invalid KV layout", ErrGemma4ContextCapacity)
	}
	for _, window := range windows {
		if window < 0 || window > nvidia.MaxDecisionAttentionTokens {
			return nil, fmt.Errorf("%w: invalid sliding window", ErrGemma4ContextCapacity)
		}
	}
	rows := make([]int, len(dims))
	for l := range rows {
		rows[l] = trunkLen
		if windows[l] > 0 {
			rows[l] = min(trunkLen, windows[l]+Gemma4PackedRows)
		}
	}
	if err := checkGemma4ArenaCapacityRows(dims, rows, branches, maxDepth); err != nil {
		return nil, err
	}
	a := &gemma4NVIDIAKVArena{trunkK: make([]*nvidia.Buffer, len(dims)), trunkV: make([]*nvidia.Buffer, len(dims)), suffixK: make([]*nvidia.Buffer, len(dims)), suffixV: make([]*nvidia.Buffer, len(dims)), kvDim: append([]int(nil), dims...), trunkBase: make([]int, len(dims)), trunkRows: rows, windows: append([]int(nil), windows...), branches: branches, trunkLen: trunkLen, trunkUsed: trunkLen, suffixCap: maxDepth}
	ok := false
	defer func() {
		if !ok {
			a.free()
		}
	}()
	for l, dim := range dims {
		if dim == 0 {
			continue
		}
		var err error
		a.trunkK[l], err = nvidia.Malloc(rows[l] * dim)
		if err != nil {
			return nil, err
		}
		a.trunkV[l], err = nvidia.Malloc(rows[l] * dim)
		if err != nil {
			return nil, err
		}
		a.suffixK[l], err = nvidia.Malloc(branches * maxDepth * dim)
		if err != nil {
			return nil, err
		}
		a.suffixV[l], err = nvidia.Malloc(branches * maxDepth * dim)
		if err != nil {
			return nil, err
		}
	}
	ok = true
	return a, nil
}

func (a *gemma4NVIDIAKVArena) cloneTrunk(newTrunkLen, branches, suffixCap int) (*gemma4NVIDIAKVArena, error) {
	if a == nil || newTrunkLen < a.trunkLen {
		return nil, fmt.Errorf("invalid NVIDIA trunk clone")
	}
	out, err := allocGemma4NVIDIAKVArenaWindowed(a.kvDim, newTrunkLen, branches, suffixCap, a.windows)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			out.free()
		}
	}()
	for l, dim := range a.kvDim {
		if dim == 0 {
			continue
		}
		out.trunkBase[l] = a.trunkBase[l]
		n := uint64(min(a.trunkRows[l], out.trunkRows[l]) * dim * 4)
		if err := nvidia.CopyDtoD(out.trunkK[l].Ptr, a.trunkK[l].Ptr, n); err != nil {
			return nil, err
		}
		if err := nvidia.CopyDtoD(out.trunkV[l].Ptr, a.trunkV[l].Ptr, n); err != nil {
			return nil, err
		}
	}
	ok = true
	return out, nil
}

func allocGemma4NVIDIAKVArenaShape(dims []int, trunkLen, branches, maxDepth int) (*gemma4NVIDIAKVArena, error) {
	return allocGemma4NVIDIAKVArenaWindowed(dims, trunkLen, branches, maxDepth, make([]int, len(dims)))
}

func (a *gemma4NVIDIAKVArena) free() {
	if a == nil {
		return
	}
	for _, set := range [][]*nvidia.Buffer{a.trunkK, a.trunkV, a.suffixK, a.suffixV} {
		for _, b := range set {
			if b != nil {
				b.Free()
			}
		}
	}
	a.trunkK = nil
	a.trunkV = nil
	a.suffixK = nil
	a.suffixV = nil
}
func (a *gemma4NVIDIAKVArena) appendTrunkRows(layer, pos, rows int, k, v *nvidia.Buffer) error {
	if a == nil || layer < 0 || layer >= len(a.trunkK) || a.trunkK[layer] == nil || pos < 0 || rows <= 0 || pos+rows > a.trunkLen {
		return fmt.Errorf("invalid NVIDIA trunk KV append")
	}
	dim := a.kvDim[layer]
	if k == nil || v == nil || k.Size < rows*dim*4 || v.Size < rows*dim*4 {
		return fmt.Errorf("invalid NVIDIA trunk KV append shape")
	}
	if err := a.prepareTrunkWrite(layer, pos, rows); err != nil {
		return err
	}
	dst := nvidia.CUdeviceptr((pos - a.trunkBase[layer]) * dim * 4)
	n := uint64(rows * dim * 4)
	if err := nvidia.CopyDtoD(a.trunkK[layer].Ptr+dst, k.Ptr, n); err != nil {
		return err
	}
	return nvidia.CopyDtoD(a.trunkV[layer].Ptr+dst, v.Ptr, n)
}

func (a *gemma4NVIDIAKVArena) appendTrunkRow(layer, pos int, k, v *nvidia.Buffer) error {
	return a.appendTrunkRows(layer, pos, 1, k, v)
}

func (a *gemma4NVIDIAKVArena) appendRows(layer, pos int, active []int, k, v *nvidia.Buffer) error {
	if a == nil || layer < 0 || layer >= len(a.suffixK) || a.suffixK[layer] == nil || len(active) == 0 {
		return fmt.Errorf("invalid NVIDIA KV append")
	}
	dim := a.kvDim[layer]
	suffixPos := pos - a.trunkUsed
	if suffixPos < 0 || suffixPos >= a.suffixCap || k == nil || v == nil || k.Size < len(active)*dim*4 || v.Size < len(active)*dim*4 {
		return fmt.Errorf("invalid NVIDIA KV append shape")
	}
	for row, branch := range active {
		if branch < 0 || branch >= a.branches {
			return fmt.Errorf("active branch %d out of range", branch)
		}
		dst := (branch*a.suffixCap + suffixPos) * dim * 4
		src := row * dim * 4
		if err := nvidia.CopyDtoD(a.suffixK[layer].Ptr+nvidia.CUdeviceptr(dst), k.Ptr+nvidia.CUdeviceptr(src), uint64(dim*4)); err != nil {
			return err
		}
		if err := nvidia.CopyDtoD(a.suffixV[layer].Ptr+nvidia.CUdeviceptr(dst), v.Ptr+nvidia.CUdeviceptr(src), uint64(dim*4)); err != nil {
			return err
		}
	}
	return nil
}
func (a *gemma4NVIDIAKVArena) activeIndexBuffer(active []int) (*nvidia.Buffer, error) {
	b, err := nvidia.Malloc(len(active))
	if err != nil {
		return nil, err
	}
	ids := make([]uint32, len(active))
	for i, v := range active {
		if v < 0 || v >= a.branches {
			b.Free()
			return nil, fmt.Errorf("active branch %d out of range", v)
		}
		ids[i] = uint32(v)
	}
	if err := b.UploadUint32(ids); err != nil {
		b.Free()
		return nil, err
	}
	return b, nil
}
