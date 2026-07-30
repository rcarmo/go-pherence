package nvidia

import (
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/internal/checked"
)

const attentionSplitKVChunkSize = 256

var (
	attnSplitKVPartialFn CUfunction
	attnSplitKVMergeFn   CUfunction

	defaultAttentionSplitKVCandidate = NewAttentionSplitKVCandidate()
)

// AttentionSplitKVCandidate is an opt-in decode-attention candidate that keeps
// score materialization chunk-local. It is not part of the production
// DevAttention dispatcher.
//
// The caller must not call RunOK again, or Free the candidate, until prior
// launches that used the candidate have completed on the default stream.
// Launches from one RunOK call are serialized together so the shared partial
// buffers cannot be interleaved across concurrent callers.
type AttentionSplitKVCandidate struct {
	mu sync.Mutex

	partialMax *Buffer
	partialSum *Buffer
	partialOut *Buffer

	headChunkCap  int
	partialOutCap int
}

func NewAttentionSplitKVCandidate() *AttentionSplitKVCandidate {
	return &AttentionSplitKVCandidate{}
}

func (c *AttentionSplitKVCandidate) Free() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.partialMax != nil {
		c.partialMax.Free()
		c.partialMax = nil
	}
	if c.partialSum != nil {
		c.partialSum.Free()
		c.partialSum = nil
	}
	if c.partialOut != nil {
		c.partialOut.Free()
		c.partialOut = nil
	}
	c.headChunkCap = 0
	c.partialOutCap = 0
}

func shutdownAttentionSplitKVCandidate() {
	if defaultAttentionSplitKVCandidate != nil {
		defaultAttentionSplitKVCandidate.Free()
	}
}

// DevAttentionSplitKVReady reports whether the opt-in split-KV candidate is
// available.
func DevAttentionSplitKVReady() bool {
	initRoPEAttn()
	return sgemmReady && attnSplitKVPartialFn != 0 && attnSplitKVMergeFn != 0
}

// DevAttentionSplitKV launches the opt-in split-KV decode-attention candidate.
func DevAttentionSplitKV(out, q, kCache, vCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) {
	_ = DevAttentionSplitKVOK(out, q, kCache, vCache, seqLen, nHeads, nKVHeads, headDim, scale)
}

// DevAttentionSplitKVOK launches the opt-in split-KV decode-attention
// candidate. Production dispatch remains on DevAttentionOK.
func DevAttentionSplitKVOK(out, q, kCache, vCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) bool {
	if defaultAttentionSplitKVCandidate == nil {
		return false
	}
	return defaultAttentionSplitKVCandidate.RunOK(out, q, kCache, vCache, seqLen, nHeads, nKVHeads, headDim, scale)
}

// RunOK launches the split-KV decode-attention candidate using reusable
// scratch buffers owned by the candidate.
func (c *AttentionSplitKVCandidate) RunOK(out, q, kCache, vCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) bool {
	initRoPEAttn()
	if c == nil {
		return false
	}
	qLen, okQ := checked.MulInt(nHeads, headDim)
	kvDim, okKVDim := checked.MulInt(nKVHeads, headDim)
	cacheLen, okCache := checked.MulInt(seqLen, kvDim)
	nChunks := attentionSplitKVChunkCount(seqLen)
	headChunks, okHeadChunks := checked.MulInt(nHeads, nChunks)
	partialOutElems, okPartialOut := checked.MulInt(headChunks, headDim)
	if !DevAttentionSplitKVReady() || !fitsUint32(seqLen) || !fitsUint32(nHeads) || !fitsUint32(nKVHeads) || !fitsUint32(headDim) || !fitsUint32(nChunks) || seqLen <= 0 || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 || headDim > 512 || nHeads%nKVHeads != 0 || !okQ || !okKVDim || !okCache || !okHeadChunks || !okPartialOut || out == nil || q == nil || kCache == nil || vCache == nil || out.n < qLen || q.n < qLen || kCache.n < cacheLen || vCache.n < cacheLen || !tryGPU(out, q, kCache, vCache) {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := c.ensurePartialCapacityLocked(headChunks, partialOutElems); err != nil {
		return false
	}
	seqLenU := uint32(seqLen)
	nHeadsU := uint32(nHeads)
	nKVHeadsU := uint32(nKVHeads)
	headDimU := uint32(headDim)
	nChunksU := uint32(nChunks)
	if err := LaunchKernel(attnSplitKVPartialFn, nHeadsU, nChunksU, 1, attentionSplitKVChunkSize, 1, 1, 0,
		unsafe.Pointer(&q.gpu.Ptr),
		unsafe.Pointer(&kCache.gpu.Ptr),
		unsafe.Pointer(&vCache.gpu.Ptr),
		unsafe.Pointer(&c.partialMax.Ptr),
		unsafe.Pointer(&c.partialSum.Ptr),
		unsafe.Pointer(&c.partialOut.Ptr),
		unsafe.Pointer(&seqLenU),
		unsafe.Pointer(&nHeadsU),
		unsafe.Pointer(&nKVHeadsU),
		unsafe.Pointer(&headDimU),
		unsafe.Pointer(&nChunksU),
		unsafe.Pointer(&scale),
	); err != nil {
		return false
	}
	if err := LaunchKernel(attnSplitKVMergeFn, nHeadsU, 1, 1, attentionSplitKVChunkSize, 1, 1, 0,
		unsafe.Pointer(&c.partialMax.Ptr),
		unsafe.Pointer(&c.partialSum.Ptr),
		unsafe.Pointer(&c.partialOut.Ptr),
		unsafe.Pointer(&out.gpu.Ptr),
		unsafe.Pointer(&nHeadsU),
		unsafe.Pointer(&headDimU),
		unsafe.Pointer(&nChunksU),
	); err != nil {
		return false
	}
	out.dev = GPU_DEVICE
	return true
}

func (c *AttentionSplitKVCandidate) ensurePartialCapacityLocked(headChunks, partialOutElems int) error {
	if headChunks <= 0 || partialOutElems <= 0 {
		return nil
	}
	needResize := c.partialMax == nil || c.partialSum == nil || c.partialOut == nil || c.headChunkCap < headChunks || c.partialOutCap < partialOutElems
	if !needResize {
		return nil
	}
	if (c.partialMax != nil || c.partialSum != nil || c.partialOut != nil) && SyncErr() != nil {
		return SyncErr()
	}
	if c.partialMax != nil {
		c.partialMax.Free()
		c.partialMax = nil
	}
	if c.partialSum != nil {
		c.partialSum.Free()
		c.partialSum = nil
	}
	if c.partialOut != nil {
		c.partialOut.Free()
		c.partialOut = nil
	}
	partialMax, err := Malloc(headChunks)
	if err != nil {
		return err
	}
	partialSum, err := Malloc(headChunks)
	if err != nil {
		partialMax.Free()
		return err
	}
	partialOut, err := Malloc(partialOutElems)
	if err != nil {
		partialMax.Free()
		partialSum.Free()
		return err
	}
	c.partialMax = partialMax
	c.partialSum = partialSum
	c.partialOut = partialOut
	c.headChunkCap = headChunks
	c.partialOutCap = partialOutElems
	return nil
}

func attentionSplitKVChunkCount(seqLen int) int {
	if seqLen <= 0 {
		return 0
	}
	return (seqLen + attentionSplitKVChunkSize - 1) / attentionSplitKVChunkSize
}
