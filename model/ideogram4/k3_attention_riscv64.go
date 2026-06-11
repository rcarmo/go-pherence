//go:build riscv64

package ideogram4

func k3FullAttention(out, q, k, v []float32, tokens, heads, headDim int, scale float32) bool {
	if !k3Enabled() || tokens <= 0 || heads <= 0 || headDim <= 0 || len(out) < tokens*heads*headDim || len(q) < tokens*heads*headDim || len(k) < tokens*heads*headDim || len(v) < tokens*heads*headDim {
		return false
	}
	// K3 runtime seam for full non-causal attention. Current body preserves
	// scalar semantics; replace with tiled RVV/f16 attention kernel.
	scores := make([]float32, tokens)
	emb := heads * headDim
	for h := 0; h < heads; h++ {
		for ti := 0; ti < tokens; ti++ {
			qoff := ti*emb + h*headDim
			for tj := 0; tj < tokens; tj++ {
				koff := tj*emb + h*headDim
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += q[qoff+d] * k[koff+d]
				}
				scores[tj] = dot * scale
			}
			softmaxFallback(scores)
			off := ti*emb + h*headDim
			for tj := 0; tj < tokens; tj++ {
				w := scores[tj]
				voff := tj*emb + h*headDim
				for d := 0; d < headDim; d++ {
					out[off+d] += w * v[voff+d]
				}
			}
		}
	}
	return true
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
