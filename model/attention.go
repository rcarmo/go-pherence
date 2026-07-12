package model

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func attentionLogitSoftcap(cfg LlamaConfig) float32 {
	if cfg.ModelType != "gemma2" && cfg.ModelType != "gemma2_text" {
		return 0
	}
	return float32(cfg.AttentionLogitSoftcapping)
}

// attentionScale mirrors llama_model_gemma2::load_arch_hparams: Gemma2 27B
// scales by the configured Q-head width (hidden/heads), whereas 2B and 9B
// use the actual K/Q projection head width. Gemma4 supplies already-scaled Q/K.
func validateGemma2AttentionConfig(cfg LlamaConfig) error {
	if cfg.ModelType != "gemma2" && cfg.ModelType != "gemma2_text" {
		return nil
	}
	switch cfg.NumLayers {
	case 26, 42:
		return nil
	case 46:
		if cfg.NumHeads <= 0 || cfg.HiddenSize <= 0 || cfg.HiddenSize%cfg.NumHeads != 0 {
			return fmt.Errorf("Gemma2 27B attention scale requires hidden_size divisible by num_attention_heads (hidden=%d heads=%d)", cfg.HiddenSize, cfg.NumHeads)
		}
		return nil
	default:
		return fmt.Errorf("unsupported Gemma2 layer count %d: attention scale is ambiguous", cfg.NumLayers)
	}
}

func attentionScale(cfg LlamaConfig, headDim int) float32 {
	if cfg.ModelType == "gemma4_text" || cfg.ModelType == "gemma4" {
		return 1
	}
	scaleDim := headDim
	if (cfg.ModelType == "gemma2" || cfg.ModelType == "gemma2_text") && cfg.NumLayers == 46 && cfg.NumHeads > 0 && cfg.HiddenSize%cfg.NumHeads == 0 {
		scaleDim = cfg.HiddenSize / cfg.NumHeads
	}
	if scaleDim <= 0 {
		return 0
	}
	return float32(1.0 / math.Sqrt(float64(scaleDim)))
}

func gqaAttention(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int) []float32 {
	if headDim <= 0 {
		return nil
	}
	return gqaAttentionScale(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, float32(1.0/math.Sqrt(float64(headDim))))
}

func gqaAttentionScale(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	return gqaAttentionScaleSoftcap(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, 0)
}

func gqaAttentionScaleSoftcap(q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale, softcap float32) []float32 {
	if softcap <= 0 {
		out, ok := simd.GQAAttentionScaleChecked(q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
		if !ok {
			return nil
		}
		return out
	}
	out := make([]float32, numHeads*headDim)
	scores := make([]float32, seqLen)
	if !gqaAttentionScaleSoftcapInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, softcap) {
		return nil
	}
	return out
}

func gqaAttentionScaleInto(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) {
	_ = gqaAttentionScaleSoftcapInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, 0)
}

func gqaAttentionScaleSoftcapInto(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale, softcap float32) bool {
	if softcap <= 0 {
		return simd.GQAAttentionScaleTo(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
	}
	if seqLen <= 0 || numHeads <= 0 || numKVHeads <= 0 || headDim <= 0 || numHeads%numKVHeads != 0 || len(out) < numHeads*headDim || len(scores) < seqLen || len(q) < numHeads*headDim || len(kCache) < seqLen*numKVHeads*headDim || len(vCache) < seqLen*numKVHeads*headDim {
		return false
	}
	groupSize := numHeads / numKVHeads
	kvDim := numKVHeads * headDim
	invCap := float32(1) / softcap
	for head := 0; head < numHeads; head++ {
		kvHead := head / groupSize
		qRow := q[head*headDim : (head+1)*headDim]
		for token := 0; token < seqLen; token++ {
			kOff := token*kvDim + kvHead*headDim
			score := simd.Sdot(qRow, kCache[kOff:kOff+headDim]) * scale
			scores[token] = float32(math.Tanh(float64(score*invCap))) * softcap
		}
		if !simd.SoftmaxInPlace(scores[:seqLen]) {
			return false
		}
		outRow := out[head*headDim : (head+1)*headDim]
		clear(outRow)
		for token, weight := range scores[:seqLen] {
			vOff := token*kvDim + kvHead*headDim
			vRow := vCache[vOff : vOff+headDim]
			for dim := range outRow {
				outRow[dim] += weight * vRow[dim]
			}
		}
	}
	return true
}

// gqaAttentionHeadsParallel is the heads-parallel variant used by the
// sequential autoregressive decode step. It reuses the provided scores buffer
// for the serial fallback and is bit-identical to gqaAttentionScaleInto.
func gqaAttentionHeadsParallel(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) {
	gqaAttentionHeadsParallelSoftcap(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, 0)
}

func gqaAttentionHeadsParallelSoftcap(out, scores, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale, softcap float32) {
	if softcap > 0 {
		_ = gqaAttentionScaleSoftcapInto(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale, softcap)
		return
	}
	_ = simd.GQAAttentionHeadsParallelTo(out, scores, q, kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
}
