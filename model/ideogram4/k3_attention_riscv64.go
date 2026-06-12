//go:build riscv64

package ideogram4

import (
	"math"
	"sync"

	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

// k3FullAttention computes full non-causal attention using a tiled Flash
// Attention algorithm with online softmax. It parallelizes independent
// (head, query-tile) work across X100 workers.
//
// Flash Attention tiles over K/V blocks so the full scores matrix is never
// materialized, keeping memory O(blockSize) per query instead of O(tokens).
// This dramatically improves cache locality for large token counts.
func k3FullAttention(out, q, k, v []float32, tokens, heads, headDim int, scale float32) bool {
	if !k3Enabled() || tokens <= 0 || heads <= 0 || headDim <= 0 || len(out) < tokens*heads*headDim || len(q) < tokens*heads*headDim || len(k) < tokens*heads*headDim || len(v) < tokens*heads*headDim {
		return false
	}
	emb := heads * headDim
	// Block size for K/V tiling. 32 keeps one K/V tile in L1 cache:
	// 32 * 128 * 4 bytes = 16KB per tile, well within K3's 32KB L1D.
	const blockKV = 32
	jobs := tokens * heads
	nw := k3Threads()
	if nw > jobs {
		nw = jobs
	}
	if nw <= 1 || jobs < 4 {
		k3FlashAttentionWork(out, q, k, v, tokens, heads, headDim, emb, scale, blockKV, 0, jobs)
		return true
	}
	var wg sync.WaitGroup
	wg.Add(nw)
	for wid := 0; wid < nw; wid++ {
		wid := wid
		go func() {
			defer wg.Done()
			start := wid * jobs / nw
			end := (wid + 1) * jobs / nw
			k3FlashAttentionWork(out, q, k, v, tokens, heads, headDim, emb, scale, blockKV, start, end)
		}()
	}
	wg.Wait()
	return true
}

func k3FlashAttentionWork(out, q, k, v []float32, tokens, heads, headDim, emb int, scale float32, blockKV, startJob, endJob int) {
	scores := make([]float32, blockKV)
	for job := startJob; job < endJob; job++ {
		h := job / tokens
		ti := job - h*tokens
		qoff := ti*emb + h*headDim
		qRow := q[qoff : qoff+headDim]
		ooff := ti*emb + h*headDim
		oRow := out[ooff : ooff+headDim]
		// Clear output
		for d := range oRow {
			oRow[d] = 0
		}
		// Online softmax state
		runningMax := float32(-math.MaxFloat32)
		runningSum := float32(0)
		// Tile over K/V in blocks
		for kvStart := 0; kvStart < tokens; kvStart += blockKV {
			kvEnd := kvStart + blockKV
			if kvEnd > tokens {
				kvEnd = tokens
			}
			bLen := kvEnd - kvStart
			// Compute Q·K scores for this tile
			tileMax := float32(-math.MaxFloat32)
			for j := 0; j < bLen; j++ {
				tj := kvStart + j
				koff := tj*emb + h*headDim
				s := simdrt.Sdot(qRow, k[koff:koff+headDim]) * scale
				scores[j] = s
				if s > tileMax {
					tileMax = s
				}
			}
			// Online softmax update: rescale previous accumulator
			if tileMax > runningMax {
				correction := rvv.FastExp(runningMax - tileMax)
				for d := range oRow {
					oRow[d] *= correction
				}
				runningSum *= correction
				runningMax = tileMax
			}
			// Exp scores relative to running max and accumulate V
			for j := 0; j < bLen; j++ {
				tj := kvStart + j
				w := rvv.FastExp(scores[j] - runningMax)
				runningSum += w
				voff := tj*emb + h*headDim
				simdrt.Saxpy(w, v[voff:voff+headDim], oRow)
			}
		}
		// Final normalization
		if runningSum > 0 {
			inv := 1 / runningSum
			for d := range oRow {
				oRow[d] *= inv
			}
		}
	}
}

func k3QwenGQA(out, qRow, kPrefix, vPrefix []float32, seqLen, heads, kvHeads, headDim int, scale float32) bool {
	if !k3Enabled() || seqLen <= 0 || heads <= 0 || kvHeads <= 0 || headDim <= 0 || heads%kvHeads != 0 || len(out) < heads*headDim || len(qRow) < heads*headDim || len(kPrefix) < seqLen*kvHeads*headDim || len(vPrefix) < seqLen*kvHeads*headDim {
		return false
	}
	group := heads / kvHeads
	scores := make([]float32, seqLen)
	for hd := 0; hd < heads; hd++ {
		kvh := hd / group
		qoff := hd * headDim
		for tj := 0; tj < seqLen; tj++ {
			koff := tj*kvHeads*headDim + kvh*headDim
			scores[tj] = simdrt.Sdot(qRow[qoff:qoff+headDim], kPrefix[koff:koff+headDim]) * scale
		}
		softmaxFallback(scores)
		off := hd * headDim
		for d := range out[off : off+headDim] {
			out[off+d] = 0
		}
		for tj := 0; tj < seqLen; tj++ {
			w := scores[tj]
			if w == 0 {
				continue
			}
			voff := tj*kvHeads*headDim + kvh*headDim
			simdrt.Saxpy(w, vPrefix[voff:voff+headDim], out[off:off+headDim])
		}
	}
	return true
}
