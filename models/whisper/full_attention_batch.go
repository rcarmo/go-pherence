package whisper

import (
	"math"
	"sync"

	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

const fullAttentionQueryBatchDefault = 128

// fullAttentionPackedQueryBatched explores an exact-parity F32 attention path
// that packs all heads once, then evaluates each head in smaller query blocks.
// Per-head softmax semantics stay unchanged because each score row is still
// formed, normalized, and consumed independently; only the row/job scheduling
// differs from fullAttentionPerHead.
func fullAttentionPackedQueryBatched(q, k, v []float32, seqQ, seqKV, numHeads, headDim, queryBatch int) []float32 {
	dModel := numHeads * headDim
	out := make([]float32, seqQ*dModel)
	if seqQ == 0 || seqKV == 0 || numHeads == 0 || headDim == 0 {
		return out
	}
	if queryBatch < 1 || queryBatch > seqQ {
		queryBatch = seqQ
	}
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	qPacked := make([]float32, numHeads*seqQ*headDim)
	kPacked := make([]float32, numHeads*seqKV*headDim)
	vPacked := make([]float32, numHeads*seqKV*headDim)
	packAttentionHeads(qPacked, q, seqQ, numHeads, headDim)
	packAttentionHeads(kPacked, k, seqKV, numHeads, headDim)
	packAttentionHeads(vPacked, v, seqKV, numHeads, headDim)

	nw := linearWorkers
	if nw > numHeads {
		nw = numHeads
	}
	if nw < 1 {
		nw = 1
	}

	qHeadStride := seqQ * headDim
	kvHeadStride := seqKV * headDim
	work := func(hStart, hEnd int) {
		scores := make([]float32, queryBatch*seqKV)
		outh := make([]float32, queryBatch*headDim)
		for h := hStart; h < hEnd; h++ {
			hOff := h * headDim
			qh := qPacked[h*qHeadStride : (h+1)*qHeadStride]
			kh := kPacked[h*kvHeadStride : (h+1)*kvHeadStride]
			vh := vPacked[h*kvHeadStride : (h+1)*kvHeadStride]
			for q0 := 0; q0 < seqQ; q0 += queryBatch {
				qm := queryBatch
				if q0+qm > seqQ {
					qm = seqQ - q0
				}
				scoreBlock := scores[:qm*seqKV]
				clear(scoreBlock)
				if !simdrt.SgemmNTTo(scoreBlock, qh[q0*headDim:], kh, qm, seqKV, headDim, scale, headDim, headDim, seqKV) {
					attnScalarHead(scoreBlock, qh[q0*headDim:], kh, qm, seqKV, headDim, scale)
				}
				for tq := 0; tq < qm; tq++ {
					softmax(scoreBlock[tq*seqKV : (tq+1)*seqKV])
				}
				outBlock := outh[:qm*headDim]
				clear(outBlock)
				if !simdrt.SgemmNNTo(outBlock, scoreBlock, vh, qm, headDim, seqKV, 1.0, seqKV, headDim, headDim) {
					attnScalarAV(outBlock, scoreBlock, vh, qm, seqKV, headDim)
				}
				for tq := 0; tq < qm; tq++ {
					copy(out[(q0+tq)*dModel+hOff:(q0+tq)*dModel+hOff+headDim], outBlock[tq*headDim:(tq+1)*headDim])
				}
			}
		}
	}

	if nw <= 1 {
		work(0, numHeads)
		return out
	}
	chunk := (numHeads + nw - 1) / nw
	var wg sync.WaitGroup
	for hs := 0; hs < numHeads; hs += chunk {
		he := hs + chunk
		if he > numHeads {
			he = numHeads
		}
		wg.Add(1)
		go func(hs, he int) {
			defer wg.Done()
			work(hs, he)
		}(hs, he)
	}
	wg.Wait()
	return out
}

func packAttentionHeads(dst, src []float32, seqLen, numHeads, headDim int) {
	dModel := numHeads * headDim
	for t := 0; t < seqLen; t++ {
		srcRow := src[t*dModel : (t+1)*dModel]
		for h := 0; h < numHeads; h++ {
			dstOff := (h*seqLen + t) * headDim
			copy(dst[dstOff:dstOff+headDim], srcRow[h*headDim:(h+1)*headDim])
		}
	}
}
