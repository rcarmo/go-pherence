//go:build riscv64

package ideogram4

import (
	"math"
	"sync"

	simdrt "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// k3FullAttention computes full non-causal attention using X100 workers and the
// SIMD/RVV dot-dispatch path. It parallelizes independent (head, query-token)
// rows and keeps one score scratch per worker. This replaces the original K3
// seam that preserved scalar semantics in a single goroutine.
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
		maxScore := float32(-math.MaxFloat32)
		for tj := 0; tj < tokens; tj++ {
			koff := tj*emb + h*headDim
			s := simdrt.Sdot(q[qoff:qoff+headDim], k[koff:koff+headDim]) * scale
			scores[tj] = s
			if s > maxScore {
				maxScore = s
			}
		}
		var sum float32
		for tj := 0; tj < tokens; tj++ {
			e := float32(math.Exp(float64(scores[tj] - maxScore)))
			scores[tj] = e
			sum += e
		}
		if sum != 0 {
			inv := 1 / sum
			for tj := 0; tj < tokens; tj++ {
				scores[tj] *= inv
			}
		}
		off := ti*emb + h*headDim
		for d := 0; d < headDim; d++ {
			var acc float32
			for tj := 0; tj < tokens; tj++ {
				acc += scores[tj] * v[tj*emb+h*headDim+d]
			}
			out[off+d] = acc
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
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += qRow[qoff+d] * kPrefix[koff+d]
			}
			scores[tj] = dot * scale
		}
		softmaxFallback(scores)
		off := hd * headDim
		for tj := 0; tj < seqLen; tj++ {
			w := scores[tj]
			voff := tj*kvHeads*headDim + kvh*headDim
			for d := 0; d < headDim; d++ {
				out[off+d] += w * vPrefix[voff+d]
			}
		}
	}
	return true
}
