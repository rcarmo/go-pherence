package model

// Batched prefill: process all prompt tokens through the model at once.
// Reads each weight matrix once instead of B times → B× faster prefill.
//
// For a 6-token prompt with 28 layers:
//   Sequential: 28 × 6 × 7 GEMV = 1176 weight reads
//   Batched:    28 × 7 GEMM     = 196 weight reads (6× fewer)

import (
	"fmt"
	"os"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

// prefillGPU processes all prompt tokens through the model in one batched pass.
// Returns the hidden state for the last token, with KV cache filled for all positions.
func (g *GPUModel) prefillGPU(tokenIDs []int) []float32 {
	if g == nil || g.CPU == nil || g.CPU.EmbedTokens == nil {
		return nil
	}
	cfg := g.CPU.Config
	h := cfg.HiddenSize
	numHeads := cfg.NumHeads
	numKVHeads := cfg.NumKVHeads
	B := len(tokenIDs)
	m := g.CPU

	if B <= 1 {
		prefillDebugf("[prefill] skip: batch=%d\n", B)
		return nil
	}
	if cfg.NumExperts > 0 || cfg.ModelType == "qwen3_moe" {
		prefillDebugf("[prefill] skip: MoE prefill is not implemented for model_type=%s experts=%d\n", cfg.ModelType, cfg.NumExperts)
		return nil
	}
	if h <= 0 || numHeads <= 0 || numKVHeads <= 0 || cfg.Intermediate <= 0 {
		prefillDebugf("[prefill] skip: invalid dims h=%d heads=%d kvHeads=%d intermediate=%d\n", h, numHeads, numKVHeads, cfg.Intermediate)
		return nil // fall back to sequential
	}
	if !nvidia.BatchGEMMReady() {
		prefillDebugf("[prefill] skip: batch GEMM kernels unavailable\n")
		return nil
	}
	headDim, err := m.LayerHeadDim(0)
	if err != nil || headDim <= 0 {
		return nil
	}
	qDim := headDim * numHeads
	// This batched layout uses one set of fixed-width buffers. Reject mixed
	// per-layer attention widths rather than corrupting slices or KV offsets.
	for l := range g.Layers {
		layerHeadDim, layerErr := m.LayerHeadDim(l)
		if layerErr != nil || layerHeadDim != headDim {
			prefillDebugf("[prefill] skip: mixed head dim at layer=%d got=%d want=%d\n", l, layerHeadDim, headDim)
			return nil
		}
	}
	maxInt := int(^uint(0) >> 1)
	if numKVHeads > maxInt/headDim || B > maxInt/h || B > maxInt/cfg.Intermediate {
		return nil
	}
	kvDim := headDim * numKVHeads
	if B > maxInt/kvDim {
		return nil
	}
	if B > maxInt/qDim {
		return nil
	}
	defaultScale := attentionScale(cfg, headDim)

	prefillDebugf("[prefill] batch=%d tokens, %d layers\n", B, len(g.Layers))

	// Allocate batch buffers: [B × dim]
	bHidden := nvidia.NewDevBuf(B * h)
	bNormed := nvidia.NewDevBuf(B * h)
	bQ := nvidia.NewDevBuf(B * qDim)
	bK := nvidia.NewDevBuf(B * kvDim)
	bV := nvidia.NewDevBuf(B * kvDim)
	bAttnOut := nvidia.NewDevBuf(B * qDim)
	bOOut := nvidia.NewDevBuf(B * h)
	bGate := nvidia.NewDevBuf(B * cfg.Intermediate)
	bUp := nvidia.NewDevBuf(B * cfg.Intermediate)
	bDown := nvidia.NewDevBuf(B * h)
	bResidual := nvidia.NewDevBuf(B * h)
	defer bHidden.Free()
	defer bNormed.Free()
	defer bQ.Free()
	defer bK.Free()
	defer bV.Free()
	defer bAttnOut.Free()
	defer bOOut.Free()
	defer bGate.Free()
	defer bUp.Free()
	defer bDown.Free()
	defer bResidual.Free()
	for _, buf := range []*nvidia.DevBuf{bNormed, bQ, bK, bV, bAttnOut, bOOut, bGate, bUp, bDown, bResidual} {
		if err := buf.ToGPU(); err != nil {
			return nil
		}
	}

	// Prepare all prompt rows. Multimodal callers supply exact embeddings;
	// ordinary text callers gather rows from the token table.
	hd := bHidden.Data()
	if g.promptEmbeddings != nil {
		if len(g.promptEmbeddings) != B*h {
			return nil
		}
		copy(hd, g.promptEmbeddings)
	} else {
		embData := m.EmbedTokens.Data()
		if cfg.VocabSize <= 0 || len(embData) < cfg.VocabSize*h {
			return nil
		}
		for i, tokID := range tokenIDs {
			if tokID < 0 || tokID >= cfg.VocabSize {
				return nil
			}
			copy(hd[i*h:(i+1)*h], embData[tokID*h:(tokID+1)*h])
		}
	}
	bHidden.MarkDirty()
	if err := bHidden.ToGPU(); err != nil {
		return nil
	}

	// Process each layer with batched ops
	for l := 0; l < len(g.Layers); l++ {
		layer := &g.Layers[l]

		// Sync every 20 layers to prevent command queue overflow
		if l > 0 && l%20 == 0 {
			nvidia.Sync()
		}

		// Save residual: bResidual = bHidden
		nvidia.DevCopy(bResidual, bHidden)

		// RMSNorm each row: bNormed[b] = rmsNorm(bHidden[b])
		// For now, do per-row RMSNorm on GPU (each row independently)
		for b := 0; b < B; b++ {
			hSlice := bHidden.Slice(b*h, h)
			nSlice := bNormed.Slice(b*h, h)
			nvidia.DevRMSNorm(nSlice, hSlice, layer.InputNorm, float32(cfg.RMSNormEps))
		}

		// Batched Q/K/V projections: read weights once for all B tokens
		if layer.QWg != nil {
			nvidia.PrefetchWeights(layer.OWg, layer.GateWg, layer.UpWg, layer.DownWg)
			nvidia.GemmQ4(bQ, bNormed, layer.QWg, B)
			nvidia.GemmQ4(bK, bNormed, layer.KWg, B)
			if cfg.AttentionKEqV && (layer.VWg == nil || layer.VWg == layer.KWg) {
				nvidia.DevCopy(bV, bK)
			} else if layer.VWg != nil {
				nvidia.GemmQ4(bV, bNormed, layer.VWg, B)
			} else {
				return nil
			}
			nvidia.WaitPrefetch()
		} else if layer.QW != nil && layer.KW != nil && layer.VW != nil {
			if nvidia.Sgemm(B, qDim, h, 1, bNormed.GPUBuffer(), layer.QW.GPUBuffer(), bQ.GPUBuffer()) != nil ||
				nvidia.Sgemm(B, kvDim, h, 1, bNormed.GPUBuffer(), layer.KW.GPUBuffer(), bK.GPUBuffer()) != nil ||
				nvidia.Sgemm(B, kvDim, h, 1, bNormed.GPUBuffer(), layer.VW.GPUBuffer(), bV.GPUBuffer()) != nil {
				return nil
			}
		} else {
			return nil
		}

		// Bias (per-row broadcast)
		if layer.QB != nil {
			for b := 0; b < B; b++ {
				nvidia.DevAdd(bQ.Slice(b*qDim, qDim), bQ.Slice(b*qDim, qDim), layer.QB)
				nvidia.DevAdd(bK.Slice(b*kvDim, kvDim), bK.Slice(b*kvDim, kvDim), layer.KB)
				nvidia.DevAdd(bV.Slice(b*kvDim, kvDim), bV.Slice(b*kvDim, kvDim), layer.VB)
			}
		}

		// RoPE + KV cache + Attention (per token — needs causal masking)
		for b := 0; b < B; b++ {
			pos := b
			seqLen := b + 1

			qSlice := bQ.Slice(b*qDim, qDim)
			kSlice := bK.Slice(b*kvDim, kvDim)
			vSlice := bV.Slice(b*kvDim, kvDim)

			// Qwen3 applies RMSNorm independently to every Q/K head before RoPE.
			if layer.QNorm != nil {
				for head := 0; head < numHeads; head++ {
					hs := qSlice.Slice(head*headDim, headDim)
					nvidia.DevRMSNorm(hs, hs, layer.QNorm, float32(cfg.RMSNormEps))
				}
			}
			if layer.KNorm != nil {
				for head := 0; head < numKVHeads; head++ {
					hs := kSlice.Slice(head*headDim, headDim)
					nvidia.DevRMSNorm(hs, hs, layer.KNorm, float32(cfg.RMSNormEps))
				}
			}

			// RoPE
			ropePtr := (*nvidia.Buffer)(nil)
			if g.ropeCosSin != nil {
				ropePtr = g.ropeCosSin.GPUPtr()
			}
			freqs := m.ensureRoPE(pos)
			if ropePtr != nil {
				if !nvidia.DevRoPE(qSlice, g.ropeCosSin, pos, numHeads, headDim) {
					qd := qSlice.Data()
					applyRoPE(qd, freqs, pos, numHeads, headDim)
					qSlice.MarkDirty()
				}
				if !nvidia.DevRoPE(kSlice, g.ropeCosSin, pos, numKVHeads, headDim) {
					kd := kSlice.Data()
					applyRoPE(kd, freqs, pos, numKVHeads, headDim)
					kSlice.MarkDirty()
				}
			} else {
				qd := qSlice.Data()
				kd := kSlice.Data()
				applyRoPE(qd, freqs, pos, numHeads, headDim)
				applyRoPE(kd, freqs, pos, numKVHeads, headDim)
				qSlice.MarkDirty()
				kSlice.MarkDirty()
			}

			// KV cache append
			var kvKPtr, kvVPtr, kPtr, vPtr *nvidia.Buffer
			if g.kvGPU_K[l] != nil {
				kvKPtr = g.kvGPU_K[l].GPUPtr()
			}
			if g.kvGPU_V[l] != nil {
				kvVPtr = g.kvGPU_V[l].GPUPtr()
			}
			kPtr = kSlice.GPUPtr()
			vPtr = vSlice.GPUPtr()
			if kvKPtr != nil && kvVPtr != nil && kPtr != nil && vPtr != nil {
				kvBytes, kOff, ok := kvCopyByteRange(pos, kvDim, kvKPtr.Size)
				_, _, okV := kvCopyByteRange(pos, kvDim, kvVPtr.Size)
				if !ok || !okV || kPtr.Size < int(kvBytes) || vPtr.Size < int(kvBytes) {
					return nil
				}
				if err := nvidia.CopyDtoD(kvKPtr.Ptr+kOff, kPtr.Ptr, kvBytes); err != nil {
					return nil
				}
				if err := nvidia.CopyDtoD(kvVPtr.Ptr+kOff, vPtr.Ptr, kvBytes); err != nil {
					return nil
				}
			}

			// Attention
			aSlice := bAttnOut.Slice(b*qDim, qDim)
			if g.kvGPU_K[l] != nil {
				nvidia.DevAttention(aSlice, qSlice, g.kvGPU_K[l], g.kvGPU_V[l], seqLen, numHeads, numKVHeads, headDim, defaultScale)
			}
		}

		// Batched O projection
		if layer.OWg != nil {
			nvidia.GemmQ4(bOOut, bAttnOut, layer.OWg, B)
		} else if layer.OW != nil {
			if nvidia.Sgemm(B, h, qDim, 1, bAttnOut.GPUBuffer(), layer.OW.GPUBuffer(), bOOut.GPUBuffer()) != nil {
				return nil
			}
		} else {
			return nil
		}

		// Residual add: bHidden = bResidual + bOOut
		nvidia.DevAdd(bHidden, bResidual, bOOut)

		// Post-attention norm (per row)
		for b := 0; b < B; b++ {
			hSlice := bHidden.Slice(b*h, h)
			nSlice := bNormed.Slice(b*h, h)
			nvidia.DevRMSNorm(nSlice, hSlice, layer.PostNorm, float32(cfg.RMSNormEps))
		}

		// Batched MLP
		if layer.GateWg != nil {
			nvidia.PrefetchWeights(nil) // prefetch next layer if exists
			if l+1 < len(g.Layers) {
				next := &g.Layers[l+1]
				nvidia.PrefetchWeights(next.QWg, next.KWg, next.VWg)
			}
			nvidia.GemmQ4(bGate, bNormed, layer.GateWg, B)
			nvidia.GemmQ4(bUp, bNormed, layer.UpWg, B)
		} else if layer.GateW != nil && layer.UpW != nil {
			if nvidia.Sgemm(B, cfg.Intermediate, h, 1, bNormed.GPUBuffer(), layer.GateW.GPUBuffer(), bGate.GPUBuffer()) != nil ||
				nvidia.Sgemm(B, cfg.Intermediate, h, 1, bNormed.GPUBuffer(), layer.UpW.GPUBuffer(), bUp.GPUBuffer()) != nil {
				return nil
			}
		} else {
			return nil
		}

		// SiLU(gate) * up (per row)
		for b := 0; b < B; b++ {
			gSlice := bGate.Slice(b*cfg.Intermediate, cfg.Intermediate)
			uSlice := bUp.Slice(b*cfg.Intermediate, cfg.Intermediate)
			nvidia.DevSiLUMul(gSlice, gSlice, uSlice)
		}

		// Batched down projection
		if layer.DownWg != nil {
			nvidia.GemmQ4(bDown, bGate, layer.DownWg, B)
		} else if layer.DownW != nil {
			if nvidia.Sgemm(B, h, cfg.Intermediate, 1, bGate.GPUBuffer(), layer.DownW.GPUBuffer(), bDown.GPUBuffer()) != nil {
				return nil
			}
		} else {
			return nil
		}

		// Save residual for this stage
		nvidia.DevCopy(bResidual, bHidden)

		// Residual add: bHidden = bResidual + bDown
		nvidia.DevAdd(bHidden, bResidual, bDown)
	}

	// Extract last token's hidden state
	nvidia.Sync()
	hd = bHidden.Data()
	lastHidden := make([]float32, h)
	copy(lastHidden, hd[(B-1)*h:B*h])

	prefillDebugf("[prefill] done, returning hidden[%d]\n", B-1)
	return lastHidden
}

func prefillDebugf(format string, args ...any) {
	if os.Getenv("GO_PHERENCE_PREFILL_DEBUG") != "" {
		fmt.Printf(format, args...)
	}
}
