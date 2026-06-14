package simd

import (
	"runtime"
	"sync"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// GQAAttentionHeadsParallelTo computes grouped-query attention into out,
// distributing the heads across goroutines. Each head writes a disjoint slice
// of out, so there are no races and the result is bit-identical to the serial
// GQAAttentionScaleTo (per-head Sdot/softmax/VecScaleAdd in the same order).
//
// Intended for the single-query autoregressive decode step. Do NOT call it from
// inside an existing goroutine pool (e.g. batched prefill), which already
// parallelizes across tokens — use the serial path there to avoid nesting.
func GQAAttentionHeadsParallelTo(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) bool {
	if seqLen <= 0 || numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 || numHeads%numKVHeads != 0 {
		return false
	}
	h, okH := checked.MulInt(numHeads, headDim)
	kvDim, okKV := checked.MulInt(numKVHeads, headDim)
	kvTotal, okTotal := checked.MulInt(seqLen, kvDim)
	if !okH || !okKV || !okTotal || len(out) < h || len(scores) < seqLen || len(q) < h || len(kCache) < kvTotal || len(vCache) < kvTotal {
		return false
	}

	nWorkers := runtime.GOMAXPROCS(0)
	// Only parallelize when there is enough work to amortize goroutine spin-up.
	// The serial path reuses the caller-provided scores buffer (no allocation).
	if numHeads < 2 || nWorkers <= 1 || numHeads*seqLen < 4096 {
		return gqaAttentionScaleIntoSIMD(out[:h], scores[:seqLen], q[:h], kCache[:kvTotal], vCache[:kvTotal], seqLen, numHeads, numKVHeads, headDim, scale)
	}
	if nWorkers > numHeads {
		nWorkers = numHeads
	}

	out = out[:h]
	clear(out)
	headsPerKV := numHeads / numKVHeads
	var wg sync.WaitGroup
	for w := 0; w < nWorkers; w++ {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			scores := make([]float32, seqLen)
			for head := start; head < numHeads; head += nWorkers {
				kvHead := head / headsPerKV
				qHead := q[head*headDim : (head+1)*headDim]
				for t := 0; t < seqLen; t++ {
					kHead := kCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
					scores[t] = Sdot(qHead, kHead) * scale
				}
				if !SoftmaxInPlace(scores) {
					continue
				}
				outHead := out[head*headDim : (head+1)*headDim]
				for t := 0; t < seqLen; t++ {
					vHead := vCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
					VecScaleAdd(outHead, outHead, vHead, scores[t])
				}
			}
		}(w)
	}
	wg.Wait()
	return true
}
