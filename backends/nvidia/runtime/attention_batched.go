package nvidia

import (
	"fmt"
	"github.com/rcarmo/go-pherence/internal/checked"
)

// F32BatchedGQAAttention computes GQA attention for multiple query positions
// against a shared KV cache, uploading K,V only once.
// qAll is [positions, nHeads, headDim], out is [positions, nHeads, headDim].
// kCache, vCache are [seqLen, nKVHeads, headDim] (shared across positions).
func F32BatchedGQAAttention(out, qAll, kCache, vCache []float32, positions, seqLen, nHeads, nKVHeads, headDim int, scale float32) error {
	if positions <= 0 || seqLen <= 0 || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 {
		return fmt.Errorf("invalid batched GQA dims pos=%d seqLen=%d heads=%d kv=%d headDim=%d", positions, seqLen, nHeads, nKVHeads, headDim)
	}
	qLen, okQ := checked.MulInt(nHeads, headDim)
	kvDim, okKVDim := checked.MulInt(nKVHeads, headDim)
	cacheLen, okCache := checked.MulInt(seqLen, kvDim)
	totalQ, okTotalQ := checked.MulInt(positions, qLen)
	if !okQ || !okKVDim || !okCache || !okTotalQ || len(out) < totalQ || len(qAll) < totalQ || len(kCache) < cacheLen || len(vCache) < cacheLen {
		return fmt.Errorf("invalid batched GQA buffers out=%d q=%d k=%d v=%d need q/out=%d cache=%d", len(out), len(qAll), len(kCache), len(vCache), totalQ, cacheLen)
	}

	// Upload K, V caches ONCE to GPU
	kBuf := NewDevBufFrom(kCache[:cacheLen])
	vBuf := NewDevBufFrom(vCache[:cacheLen])
	defer kBuf.Free()
	defer vBuf.Free()
	if err := kBuf.ToGPU(); err != nil {
		return err
	}
	if err := vBuf.ToGPU(); err != nil {
		return err
	}

	// Process each position reusing the same K, V on GPU
	qBuf := NewDevBuf(qLen)
	outBuf := NewDevBuf(qLen)
	defer qBuf.Free()
	defer outBuf.Free()

	for pos := 0; pos < positions; pos++ {
		qOff := pos * qLen
		// Upload this position's Q
		copy(qBuf.Data()[:qLen], qAll[qOff:qOff+qLen])
		if err := qBuf.ToGPU(); err != nil {
			return err
		}
		if err := outBuf.ToGPU(); err != nil {
			return err
		}
		// Run attention kernel (K, V already on GPU)
		if !DevAttentionOK(outBuf, qBuf, kBuf, vBuf, seqLen, nHeads, nKVHeads, headDim, scale) {
			return fmt.Errorf("batched GQA attention kernel failed at pos %d", pos)
		}
		// Download output for this position
		outBuf.ToCPU()
		copy(out[qOff:qOff+qLen], outBuf.Data()[:qLen])
	}
	return nil
}

// F32BatchedCausalGQAAttention computes causal prompt self-attention for all
// query positions. K,V are uploaded once; each query position attends only to
// the prefix [0,pos]. This is a stepping stone toward a fully fused/device-
// resident prompt prefill graph while preserving causal semantics.
func F32BatchedCausalGQAAttention(out, qAll, kCache, vCache []float32, positions, nHeads, nKVHeads, headDim int, scale float32) error {
	if positions <= 0 || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 {
		return fmt.Errorf("invalid batched causal GQA dims pos=%d heads=%d kv=%d headDim=%d", positions, nHeads, nKVHeads, headDim)
	}
	qLen, okQ := checked.MulInt(nHeads, headDim)
	kvDim, okKVDim := checked.MulInt(nKVHeads, headDim)
	cacheLen, okCache := checked.MulInt(positions, kvDim)
	totalQ, okTotalQ := checked.MulInt(positions, qLen)
	if !okQ || !okKVDim || !okCache || !okTotalQ || len(out) < totalQ || len(qAll) < totalQ || len(kCache) < cacheLen || len(vCache) < cacheLen {
		return fmt.Errorf("invalid batched causal GQA buffers out=%d q=%d k=%d v=%d need q/out=%d cache=%d", len(out), len(qAll), len(kCache), len(vCache), totalQ, cacheLen)
	}

	kBuf := NewDevBufFrom(kCache[:cacheLen])
	vBuf := NewDevBufFrom(vCache[:cacheLen])
	defer kBuf.Free()
	defer vBuf.Free()
	if err := kBuf.ToGPU(); err != nil {
		return err
	}
	if err := vBuf.ToGPU(); err != nil {
		return err
	}

	qBuf := NewDevBuf(qLen)
	outBuf := NewDevBuf(qLen)
	defer qBuf.Free()
	defer outBuf.Free()
	for pos := 0; pos < positions; pos++ {
		qOff := pos * qLen
		copy(qBuf.Data()[:qLen], qAll[qOff:qOff+qLen])
		if err := qBuf.ToGPU(); err != nil {
			return err
		}
		if err := outBuf.ToGPU(); err != nil {
			return err
		}
		if !DevAttentionOK(outBuf, qBuf, kBuf, vBuf, pos+1, nHeads, nKVHeads, headDim, scale) {
			return fmt.Errorf("batched causal GQA attention kernel failed at pos %d", pos)
		}
		outBuf.ToCPU()
		copy(out[qOff:qOff+qLen], outBuf.Data()[:qLen])
	}
	return nil
}
