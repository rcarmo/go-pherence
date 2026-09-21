package nvidia

import (
	"fmt"
	"unsafe"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// --- GPU RoPE + Attention ---

var (
	ropeFn                CUfunction
	ropePartialFn         CUfunction
	ropePartialSequenceFn CUfunction
	attnCausalBatchFn     CUfunction
	attnScoreFn           CUfunction
	softmaxRowsFn         CUfunction
	attnFn                CUfunction
	ropeReady             bool
	ropePartialReady      bool
	attnScoreReady        bool
	softmaxRowsReady      bool
	attnReady             bool
)

func initRoPEAttn() { loadMegaModule() }

// DevRoPE applies rotary position embedding on GPU (in-place).
// cosSin is a precomputed [maxSeq * headDim] buffer with interleaved cos,sin pairs.
func DevRoPE(x *DevBuf, cosSin *DevBuf, pos, nHeads, headDim int) bool {
	initRoPEAttn()
	total, okTotal := checked.MulInt(nHeads, headDim)
	halfDim, okHalf := checked.MulInt(nHeads, headDim/2)
	cosNeed, okCosNeed := checked.MulInt(pos+1, headDim)
	if ropeReady && fitsUint32(pos) && nHeads > 0 && headDim > 0 && headDim%2 == 0 && okTotal && okHalf && okCosNeed && x != nil && cosSin != nil && x.n >= total && cosSin.n >= cosNeed && tryGPU(x, cosSin) {
		grid, okGrid := grid1DFor(halfDim, 256)
		if !okGrid {
			return false
		}
		p := uint32(pos)
		nh := uint32(nHeads)
		hd := uint32(headDim)
		if err := LaunchKernel(ropeFn, grid, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&x.gpu.Ptr),
			unsafe.Pointer(&cosSin.gpu.Ptr),
			unsafe.Pointer(&p),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&hd)); err == nil {
			x.dev = GPU_DEVICE
			return true
		}
	}
	return false
}

// DevRoPEPartial applies partial rotary position embedding on GPU (in-place).
// cosSin is [maxSeq * rotHalf * 2] with interleaved cos,sin pairs.
func DevRoPEPartial(x *DevBuf, cosSin *DevBuf, pos, nHeads, headDim, rotHalf int) bool {
	initRoPEAttn()
	total, okTotal := checked.MulInt(nHeads, headDim)
	totalPairs, okPairs := checked.MulInt(nHeads, rotHalf)
	posPairs, okPosPairs := checked.MulInt(pos+1, rotHalf)
	cosNeed, okCosNeed := checked.MulInt(posPairs, 2)
	if ropePartialReady && fitsUint32(pos) && nHeads > 0 && headDim > 0 && rotHalf > 0 && rotHalf <= headDim/2 && okTotal && okPairs && okPosPairs && okCosNeed && x != nil && cosSin != nil && x.n >= total && cosSin.n >= cosNeed && tryGPU(x, cosSin) {
		grid, okGrid := grid1DFor(totalPairs, 256)
		if !okGrid {
			return false
		}
		p := uint32(pos)
		nh := uint32(nHeads)
		hd := uint32(headDim)
		rh := uint32(rotHalf)
		if err := LaunchKernel(ropePartialFn, grid, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&x.gpu.Ptr),
			unsafe.Pointer(&cosSin.gpu.Ptr),
			unsafe.Pointer(&p),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&hd),
			unsafe.Pointer(&rh)); err == nil {
			x.dev = GPU_DEVICE
			return true
		}
	}
	return false
}

// RoPEPartialRowsBuffer applies the same absolute position to every row in a
// packed F32 batch without downloading the projected Q/K values.
func RoPEPartialSequenceBuffer(x, cosSin *Buffer, rows, pos0, nHeads, headDim, rotHalf int) error {
	initRoPEAttn()
	rowElems, okRow := checked.MulInt(nHeads, headDim)
	pairs, okPairs := checked.MulInt(nHeads, rotHalf)
	total, okTotal := checked.MulInt(rows, pairs)
	posPairs, okPos := checked.MulInt(pos0+rows, rotHalf)
	cosNeed, okCos := checked.MulInt(posPairs, 2)
	if !ropePartialReady || ropePartialSequenceFn == 0 || rows <= 0 || pos0 < 0 || nHeads <= 0 || headDim <= 0 || rotHalf <= 0 || rotHalf > headDim/2 || !okRow || !okPairs || !okTotal || !okPos || !okCos || x == nil || cosSin == nil || x.Ptr == 0 || cosSin.Ptr == 0 || x.Size < rows*rowElems*4 || cosSin.Size < cosNeed*4 {
		return fmt.Errorf("invalid sequence partial RoPE")
	}
	grid, ok := grid1DFor(total, 256)
	if !ok {
		return fmt.Errorf("sequence partial RoPE grid overflow")
	}
	rr, p0, nh, hd, rh := uint32(rows), uint32(pos0), uint32(nHeads), uint32(headDim), uint32(rotHalf)
	return LaunchKernel(ropePartialSequenceFn, grid, 1, 1, 256, 1, 1, 0, unsafe.Pointer(&x.Ptr), unsafe.Pointer(&cosSin.Ptr), unsafe.Pointer(&rr), unsafe.Pointer(&p0), unsafe.Pointer(&nh), unsafe.Pointer(&hd), unsafe.Pointer(&rh))
}

func CausalBatchAttentionBuffer(out, q, k, v *Buffer, rows, pos0, kvLen, window, nHeads, nKVHeads, headDim int, scale float32) error {
	qDim, okQ := checked.MulInt(nHeads, headDim)
	kvDim, okKV := checked.MulInt(nKVHeads, headDim)
	qN, okQN := checked.MulInt(rows, qDim)
	kvN, okKN := checked.MulInt(kvLen, kvDim)
	if attnCausalBatchFn == 0 || !okQ || !okKV || !okQN || !okKN || rows <= 0 || pos0 < 0 || kvLen <= 0 || kvLen > 2048 || window < 0 || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 || nHeads%nKVHeads != 0 || out == nil || q == nil || k == nil || v == nil || out.Ptr == 0 || q.Ptr == 0 || k.Ptr == 0 || v.Ptr == 0 || out.Size < qN*4 || q.Size < qN*4 || k.Size < kvN*4 || v.Size < kvN*4 {
		return fmt.Errorf("invalid causal batch attention")
	}
	rr, p0, kl, ww, hh, kk, dd := uint32(rows), uint32(pos0), uint32(kvLen), uint32(window), uint32(nHeads), uint32(nKVHeads), uint32(headDim)
	return LaunchKernel(attnCausalBatchFn, hh, rr, 1, 256, 1, 1, 2048*4, unsafe.Pointer(&q.Ptr), unsafe.Pointer(&k.Ptr), unsafe.Pointer(&v.Ptr), unsafe.Pointer(&out.Ptr), unsafe.Pointer(&rr), unsafe.Pointer(&p0), unsafe.Pointer(&kl), unsafe.Pointer(&ww), unsafe.Pointer(&hh), unsafe.Pointer(&kk), unsafe.Pointer(&dd), unsafe.Pointer(&scale))
}

func RoPEPartialRowsBuffer(x, cosSin *Buffer, rows, pos, nHeads, headDim, rotHalf int) error {
	initRoPEAttn()
	rowElems, okRow := checked.MulInt(nHeads, headDim)
	totalPairs, okPairs := checked.MulInt(nHeads, rotHalf)
	posPairs, okPos := checked.MulInt(pos+1, rotHalf)
	cosNeed, okCos := checked.MulInt(posPairs, 2)
	if !ropePartialReady || rows <= 0 || pos < 0 || nHeads <= 0 || headDim <= 0 || rotHalf <= 0 || rotHalf > headDim/2 || !okRow || !okPairs || !okPos || !okCos || x == nil || cosSin == nil || x.Ptr == 0 || cosSin.Ptr == 0 || x.Size < rows*rowElems*4 || cosSin.Size < cosNeed*4 {
		return fmt.Errorf("invalid batched partial RoPE")
	}
	for row := 0; row < rows; row++ {
		view := x.Ptr + CUdeviceptr(row*rowElems*4)
		p, nh, hd, rh := uint32(pos), uint32(nHeads), uint32(headDim), uint32(rotHalf)
		grid, ok := grid1DFor(totalPairs, 256)
		if !ok {
			return fmt.Errorf("batched partial RoPE grid overflow")
		}
		if err := LaunchKernel(ropePartialFn, grid, 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&view), unsafe.Pointer(&cosSin.Ptr), unsafe.Pointer(&p), unsafe.Pointer(&nh), unsafe.Pointer(&hd), unsafe.Pointer(&rh)); err != nil {
			return err
		}
	}
	return nil
}

// DevAttentionScores runs the score phase of GQA attention on GPU.
// out[nHeads*seqLen], q[nHeads*headDim], kCache[seqLen*kvDim]
func DevAttentionScores(out, q, kCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) bool {
	initRoPEAttn()
	qLen, okQ := checked.MulInt(nHeads, headDim)
	kvDim, okKVDim := checked.MulInt(nKVHeads, headDim)
	cacheLen, okCache := checked.MulInt(seqLen, kvDim)
	scoreLen, okScore := checked.MulInt(nHeads, seqLen)
	if attnScoreReady && fitsUint32(seqLen) && seqLen > 0 && seqLen <= 2048 && fitsUint32(nHeads) && fitsUint32(nKVHeads) && fitsUint32(headDim) && nHeads > 0 && nKVHeads > 0 && headDim > 0 && okQ && okKVDim && okCache && okScore && out != nil && q != nil && kCache != nil && out.n >= scoreLen && q.n >= qLen && kCache.n >= cacheLen && tryGPU(out, q, kCache) {
		sl := uint32(seqLen)
		nh := uint32(nHeads)
		nkv := uint32(nKVHeads)
		hd := uint32(headDim)
		if err := LaunchKernel(attnScoreFn, uint32(nHeads), 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&q.gpu.Ptr),
			unsafe.Pointer(&kCache.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&sl),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&nkv),
			unsafe.Pointer(&hd),
			unsafe.Pointer(&scale)); err == nil {
			out.dev = GPU_DEVICE
			return true
		}
	}
	return false
}

// DevSoftmaxRows runs the softmax phase over contiguous score rows.
// in/out[nRows*seqLen], one block per row.
func DevSoftmaxRows(out, in *DevBuf, nRows, seqLen int) bool {
	initRoPEAttn()
	total, okTotal := checked.MulInt(nRows, seqLen)
	if softmaxRowsReady && fitsUint32(nRows) && fitsUint32(seqLen) && nRows > 0 && seqLen > 0 && seqLen <= 2048 && okTotal && out != nil && in != nil && out.n >= total && in.n >= total && tryGPU(out, in) {
		sl := uint32(seqLen)
		if err := LaunchKernel(softmaxRowsFn, uint32(nRows), 1, 1, 256, 1, 1, 0,
			unsafe.Pointer(&in.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&sl)); err == nil {
			out.dev = GPU_DEVICE
			return true
		}
	}
	return false
}

// DevAttention runs GQA attention on GPU.
// q[nHeads*headDim], kCache/vCache[seqLen*kvDim], out[nHeads*headDim]
func DevAttention(out, q, kCache, vCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) {
	_ = DevAttentionOK(out, q, kCache, vCache, seqLen, nHeads, nKVHeads, headDim, scale)
}

// DevAttentionOK is DevAttention plus a success flag for higher-level code that
// wants to preserve an explicit CPU fallback path.
func DevAttentionOK(out, q, kCache, vCache *DevBuf, seqLen, nHeads, nKVHeads, headDim int, scale float32) bool {
	initRoPEAttn()
	// The shared-score kernel serializes softmax and is capped at 2048 keys.
	// Split-KV is measurably faster from 512 keys on the RTX 3060 and removes
	// that cap; retain the old kernel as the short-context oracle/fallback.
	if seqLen >= 512 && DevAttentionSplitKVOK(out, q, kCache, vCache, seqLen, nHeads, nKVHeads, headDim, scale) {
		return true
	}
	qLen, okQ := checked.MulInt(nHeads, headDim)
	kvDim, okKVDim := checked.MulInt(nKVHeads, headDim)
	cacheLen, okCache := checked.MulInt(seqLen, kvDim)
	if attnReady && fitsUint32(seqLen) && seqLen > 0 && seqLen <= 2048 && fitsUint32(nHeads) && fitsUint32(nKVHeads) && fitsUint32(headDim) && nHeads > 0 && nKVHeads > 0 && headDim > 0 && okQ && okKVDim && okCache && out != nil && q != nil && kCache != nil && vCache != nil && out.n >= qLen && q.n >= qLen && kCache.n >= cacheLen && vCache.n >= cacheLen && tryGPU(out, q, kCache, vCache) {
		sl := uint32(seqLen)
		nh := uint32(nHeads)
		nkv := uint32(nKVHeads)
		hd := uint32(headDim)
		// One block per query head, 256 threads per block
		if err := LaunchKernel(attnFn, uint32(nHeads), 1, 1, 256, 1, 1, 2048*4,
			unsafe.Pointer(&q.gpu.Ptr),
			unsafe.Pointer(&kCache.gpu.Ptr),
			unsafe.Pointer(&vCache.gpu.Ptr),
			unsafe.Pointer(&out.gpu.Ptr),
			unsafe.Pointer(&sl),
			unsafe.Pointer(&nh),
			unsafe.Pointer(&nkv),
			unsafe.Pointer(&hd),
			unsafe.Pointer(&scale)); err == nil {
			out.dev = GPU_DEVICE
			return true
		}
		return false
	}
	// CPU fallback in model code
	return false
}
