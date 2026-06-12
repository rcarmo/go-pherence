//go:build riscv64

package ideogram4

import (
	"math"
	"sync"

	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/backends/spacemit/rvv"
)

// k3FullAttention computes full non-causal attention using X100 workers with
// SIMD/RVV dot-dispatch for Q·K scoring and SAXPY-style V accumulation. It
// parallelizes independent (head, query-token) rows.
func k3FullAttention(out, q, k, v []float32, tokens, heads, headDim int, scale float32) bool {
	if !k3Enabled() || tokens <= 0 || heads <= 0 || headDim <= 0 || len(out) < tokens*heads*headDim || len(q) < tokens*heads*headDim || len(k) < tokens*heads*headDim || len(v) < tokens*heads*headDim {
		return false
	}
	emb := heads * headDim
	jobs := tokens * heads
	nw := k3Threads()
	if nw > jobs {
		nw = jobs
	}
	if nw <= 1 || jobs < 4 {
		k3AttentionWork(out, q, k, v, tokens, heads, headDim, emb, scale, 0, jobs, make([]float32, tokens))
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
			k3AttentionWork(out, q, k, v, tokens, heads, headDim, emb, scale, start, end, make([]float32, tokens))
		}()
	}
	wg.Wait()
	return true
}

func k3AttentionWork(out, q, k, v []float32, tokens, heads, headDim, emb int, scale float32, startJob, endJob int, scores []float32) {
	for job := startJob; job < endJob; job++ {
		h := job / tokens
		ti := job - h*tokens
		qoff := ti*emb + h*headDim
		qRow := q[qoff : qoff+headDim]

		// Q·K scoring with SIMD dot
		maxScore := float32(-math.MaxFloat32)
		for tj := 0; tj < tokens; tj++ {
			koff := tj*emb + h*headDim
			s := simdrt.Sdot(qRow, k[koff:koff+headDim]) * scale
			scores[tj] = s
			if s > maxScore {
				maxScore = s
			}
		}

		// softmax
		var sum float32
		for tj := 0; tj < tokens; tj++ {
			e := rvv.FastExp(scores[tj] - maxScore)
			scores[tj] = e
			sum += e
		}
		if sum != 0 {
			inv := 1 / sum
			for tj := 0; tj < tokens; tj++ {
				scores[tj] *= inv
			}
		}

		// V accumulation: SAXPY-style (cache-friendly, SIMD-able)
		off := ti*emb + h*headDim
		outRow := out[off : off+headDim]
		for d := range outRow {
			outRow[d] = 0
		}
		for tj := 0; tj < tokens; tj++ {
			w := scores[tj]
			if w == 0 {
				continue
			}
			voff := tj*emb + h*headDim
			simdrt.Saxpy(w, v[voff:voff+headDim], outRow)
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
