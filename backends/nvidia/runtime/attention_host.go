package nvidia

import "fmt"

// F32GQAAttention computes one causal GQA attention output for a single query
// row against prefix KV caches. q is [nHeads, headDim], k/v are
// [seqLen, nKVHeads, headDim], out is [nHeads, headDim].
func F32GQAAttention(out, q, kCache, vCache []float32, seqLen, nHeads, nKVHeads, headDim int, scale float32) error {
	if seqLen <= 0 || nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 {
		return fmt.Errorf("invalid F32 GQA dims seqLen=%d heads=%d kv=%d headDim=%d", seqLen, nHeads, nKVHeads, headDim)
	}
	qLen, okQ := checkedMulInt(nHeads, headDim)
	kvDim, okKVDim := checkedMulInt(nKVHeads, headDim)
	cacheLen, okCache := checkedMulInt(seqLen, kvDim)
	if !okQ || !okKVDim || !okCache || len(out) < qLen || len(q) < qLen || len(kCache) < cacheLen || len(vCache) < cacheLen {
		return fmt.Errorf("invalid F32 GQA buffers out=%d q=%d k=%d v=%d need out/q=%d cache=%d", len(out), len(q), len(kCache), len(vCache), qLen, cacheLen)
	}
	outBuf := NewDevBuf(qLen)
	qBuf := NewDevBufFrom(q[:qLen])
	kBuf := NewDevBufFrom(kCache[:cacheLen])
	vBuf := NewDevBufFrom(vCache[:cacheLen])
	defer outBuf.Free()
	defer qBuf.Free()
	defer kBuf.Free()
	defer vBuf.Free()
	if err := outBuf.ToGPU(); err != nil {
		return err
	}
	if err := qBuf.ToGPU(); err != nil {
		return err
	}
	if err := kBuf.ToGPU(); err != nil {
		return err
	}
	if err := vBuf.ToGPU(); err != nil {
		return err
	}
	if !DevAttentionOK(outBuf, qBuf, kBuf, vBuf, seqLen, nHeads, nKVHeads, headDim, scale) {
		return fmt.Errorf("F32 GQA attention kernel failed")
	}
	copy(out[:qLen], outBuf.Data()[:qLen])
	return nil
}
