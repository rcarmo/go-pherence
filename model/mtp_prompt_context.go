package model

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/backends/mlx"
	"github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/half"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
)

// MTPPromptContext is the real verifier-side state needed to seed a q-only
// Gemma4 MTP drafter after prompt prefill.
type MTPPromptContext struct {
	Tokens         []int
	PreviousToken  int
	Activation     []float32
	KVCacheK       [][]float32
	KVCacheV       [][]float32
	SeqLen         int
	FinalLogits    []float32
	FinalToken     int
	LogitsComputed bool
}

// BuildMTPPromptContext runs the prompt through the CPU Gemma4 MTP
// prompt/verifier layer contract, including Gemma4 per-layer inputs, and returns
// copied final activation plus float KV caches. This path intentionally tracks
// the llama.cpp/LiteRT MTP graph contract rather than ordinary Generate's
// implementation details. It is intended for MTP wiring/smokes, not as a
// high-throughput prefill implementation.
func (m *LlamaModel) BuildMTPPromptContext(tokenIDs []int) (MTPPromptContext, error) {
	if m == nil {
		return MTPPromptContext{}, fmt.Errorf("nil model")
	}
	prepared := m.prepareGenerateTokens(tokenIDs)
	if len(prepared) == 0 {
		return MTPPromptContext{}, fmt.Errorf("empty prompt")
	}
	cfg := m.Config
	h := cfg.HiddenSize
	if h <= 0 || cfg.NumLayers < 0 || len(m.Layers) < cfg.NumLayers || cfg.NumHeads <= 0 || cfg.NumKVHeads <= 0 || cfg.HeadDim <= 0 {
		return MTPPromptContext{}, fmt.Errorf("invalid model dims hidden=%d layers=%d/%d heads=%d kvHeads=%d headDim=%d", h, cfg.NumLayers, len(m.Layers), cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim)
	}
	kvCacheK := make([][]float32, cfg.NumLayers)
	kvCacheV := make([][]float32, cfg.NumLayers)
	for l := 0; l < cfg.NumLayers; l++ {
		kvDim, err := m.LayerKVDim(l)
		if err != nil {
			return MTPPromptContext{}, err
		}
		if kvDim > 0 {
			capElems, ok := checkedProduct(len(prepared), kvDim)
			if !ok {
				return MTPPromptContext{}, fmt.Errorf("KV cache capacity overflows layer=%d seq=%d kvDim=%d", l, len(prepared), kvDim)
			}
			kvCacheK[l] = make([]float32, 0, capElems)
			kvCacheV[l] = make([]float32, 0, capElems)
		}
	}
	maxHeadDim := cfg.HeadDim
	for i := range m.Layers {
		layerHeadDim, err := m.LayerHeadDim(i)
		if err != nil {
			return MTPPromptContext{}, err
		}
		if layerHeadDim > maxHeadDim {
			maxHeadDim = layerHeadDim
		}
	}
	attnScoresScratch := make([]float32, len(prepared))
	attnOutScratch := make([]float32, cfg.NumHeads*maxHeadDim)
	var finalActivation []float32
	for step, tokID := range prepared {
		hidden := make([]float32, h)
		if err := m.ScaledTokenEmbeddingInto(hidden, tokID); err != nil {
			return MTPPromptContext{}, fmt.Errorf("prompt token %d embedding: %w", step, err)
		}
		perLayerInputs, err := m.Gemma4PerLayerInputs(hidden, tokID)
		if err != nil {
			return MTPPromptContext{}, fmt.Errorf("prompt token %d per-layer inputs: %w", step, err)
		}
		for l := 0; l < cfg.NumLayers; l++ {
			var err error
			hidden, err = m.forwardMTPPromptLayer(hidden, perLayerInputs, l, step, kvCacheK, kvCacheV, attnScoresScratch, attnOutScratch)
			if err != nil {
				return MTPPromptContext{}, fmt.Errorf("prompt token %d layer %d: %w", step, l, err)
			}
		}
		if step == len(prepared)-1 {
			activation, err := m.FinishCPUActivation(hidden)
			if err != nil {
				return MTPPromptContext{}, fmt.Errorf("prompt activation finish: %w", err)
			}
			finalActivation = append([]float32(nil), activation...)
		}
	}
	return MTPPromptContext{Tokens: prepared, PreviousToken: prepared[len(prepared)-1], Activation: finalActivation, KVCacheK: kvCacheK, KVCacheV: kvCacheV, SeqLen: len(prepared), FinalToken: -1}, nil
}

func (m *LlamaModel) forwardMTPPromptLayer(hidden []float32, perLayerInputs [][]float32, layerIdx, pos int, kvCacheK, kvCacheV [][]float32, attnScoresScratch, attnOutScratch []float32) ([]float32, error) {
	return m.forwardMTPPromptLayerForRow(hidden, perLayerInputs, -1, layerIdx, pos, kvCacheK, kvCacheV, attnScoresScratch, attnOutScratch)
}

func (m *LlamaModel) forwardMTPPromptLayerForRow(hidden []float32, perLayerInputs [][]float32, traceRow, layerIdx, pos int, kvCacheK, kvCacheV [][]float32, attnScoresScratch, attnOutScratch []float32) ([]float32, error) {
	cfg := m.Config
	h := cfg.HiddenSize
	layer := &m.Layers[layerIdx]
	if len(hidden) < h || layer.InputNorm == nil || layer.PostNorm == nil {
		return nil, fmt.Errorf("invalid hidden/norms")
	}
	residual := make([]float32, h)
	copy(residual, hidden[:h])
	layerHeadDim, err := m.LayerHeadDim(layerIdx)
	if err != nil {
		return nil, err
	}
	layerKVHeads := gemmacfg.LayerKVHeads(cfg, layerIdx)
	qDim, ok := checkedProduct(cfg.NumHeads, layerHeadDim)
	layerKVDim, okKV := checkedProduct(layerKVHeads, layerHeadDim)
	if layerHeadDim <= 0 || !ok || !okKV {
		return nil, fmt.Errorf("invalid attention dims")
	}
	if cfg.ModelType == "gemma3_text" {
		simd.RMSNormBF16(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))
	} else {
		rmsNormInPlace(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))
	}
	traceMTPVerifierLayer0Internal("attn_norm", layerIdx, pos, hidden)
	traceMTPSummary("attn_norm", traceRow, layerIdx, pos, hidden)
	q := make([]float32, qDim)
	if layer.QWq != nil {
		if !m.mvQ(q, hidden, layer.QWq) {
			return nil, fmt.Errorf("layer %d quantized Q projection failed", layerIdx)
		}
	} else if layer.QWm != nil {
		if !mlx.GemvTo(q, hidden, layer.QWm) {
			return nil, fmt.Errorf("layer %d MLX Q projection failed", layerIdx)
		}
	} else if layer.QWGGUF != nil {
		if !gemvGGUFTo(q, hidden, layer.QWGGUF, h, qDim) {
			return nil, fmt.Errorf("layer %d GGUF Q projection failed", layerIdx)
		}
	} else {
		m.mv(q, hidden, layer.QW.Data(), h, qDim)
	}
	traceMTPVerifierLayer0Internal("q_proj", layerIdx, pos, q)
	traceMTPSummary("q_proj", traceRow, layerIdx, pos, q)
	var k, v []float32
	if layer.HasKV {
		k = make([]float32, layerKVDim)
		v = make([]float32, layerKVDim)
		if layer.KWq != nil {
			if !m.mvQ(k, hidden, layer.KWq) {
				return nil, fmt.Errorf("layer %d quantized K projection failed", layerIdx)
			}
			if cfg.AttentionKEqV && (layer.VWq == nil || layer.VWq == layer.KWq) {
				copy(v, k)
			} else if layer.VWq != nil {
				if !m.mvQ(v, hidden, layer.VWq) {
					return nil, fmt.Errorf("layer %d quantized V projection failed", layerIdx)
				}
			} else {
				return nil, fmt.Errorf("layer %d missing quantized V projection", layerIdx)
			}
		} else if layer.KWm != nil {
			if !mlx.GemvTo(k, hidden, layer.KWm) {
				return nil, fmt.Errorf("layer %d MLX K projection failed", layerIdx)
			}
			if cfg.AttentionKEqV && (layer.VWm == nil || layer.VWm == layer.KWm) {
				copy(v, k)
			} else if layer.VWm != nil {
				if !mlx.GemvTo(v, hidden, layer.VWm) {
					return nil, fmt.Errorf("layer %d MLX V projection failed", layerIdx)
				}
			} else {
				return nil, fmt.Errorf("layer %d missing MLX V projection", layerIdx)
			}
		} else if layer.KWGGUF != nil {
			if !gemvGGUFTo(k, hidden, layer.KWGGUF, h, layerKVDim) {
				return nil, fmt.Errorf("layer %d GGUF K projection failed", layerIdx)
			}
			if cfg.AttentionKEqV && (layer.VWGGUF == nil || layer.VWGGUF == layer.KWGGUF) {
				copy(v, k)
			} else if layer.VWGGUF != nil {
				if !gemvGGUFTo(v, hidden, layer.VWGGUF, h, layerKVDim) {
					return nil, fmt.Errorf("layer %d GGUF V projection failed", layerIdx)
				}
			} else {
				return nil, fmt.Errorf("layer %d missing GGUF V projection", layerIdx)
			}
		} else {
			if layer.KW == nil {
				return nil, fmt.Errorf("layer %d missing K projection", layerIdx)
			}
			m.mv(k, hidden, layer.KW.Data(), h, layerKVDim)
			if cfg.AttentionKEqV && (layer.VW == nil || layer.VW == layer.KW) {
				copy(v, k)
			} else if layer.VW != nil {
				m.mv(v, hidden, layer.VW.Data(), h, layerKVDim)
			} else {
				return nil, fmt.Errorf("layer %d missing V projection", layerIdx)
			}
		}
		traceMTPVerifierLayer0Internal("k_proj", layerIdx, pos, k)
		traceMTPSummary("k_proj", traceRow, layerIdx, pos, k)
		traceMTPVerifierLayer0Internal("v_proj", layerIdx, pos, v)
		traceMTPSummary("v_proj", traceRow, layerIdx, pos, v)
	}
	if cfg.ModelType == "gemma3_text" {
		simd.ToBF16(q)
		if k != nil {
			simd.ToBF16(k)
			simd.ToBF16(v)
		}
	}
	if layer.QB != nil {
		simd.VecAdd(q, q, layer.QB.Data())
		if k != nil {
			simd.VecAdd(k, k, layer.KB.Data())
			simd.VecAdd(v, v, layer.VB.Data())
		}
	}
	normFn := rmsNormInPlace
	if cfg.ModelType == "gemma3_text" {
		normFn = rmsNormBF16
	}
	if cfg.ModelType == "gemma4_text" && v != nil {
		eps := float32(cfg.RMSNormEps)
		for head := 0; head < layerKVHeads; head++ {
			simd.RMSNormNoScale(v[head*layerHeadDim:(head+1)*layerHeadDim], eps)
		}
	} else if layer.VNorm != nil && v != nil {
		vnorm := layer.VNorm.Data()
		for head := 0; head < layerKVHeads; head++ {
			normFn(v[head*layerHeadDim:(head+1)*layerHeadDim], vnorm, float32(cfg.RMSNormEps))
		}
	}
	if layer.QNorm != nil {
		qNorm := layer.QNorm.Data()
		for head := 0; head < cfg.NumHeads; head++ {
			normFn(q[head*layerHeadDim:(head+1)*layerHeadDim], qNorm, float32(cfg.RMSNormEps))
		}
		if k != nil {
			if layer.KNorm == nil {
				return nil, fmt.Errorf("missing K norm")
			}
			kNorm := layer.KNorm.Data()
			for head := 0; head < layerKVHeads; head++ {
				normFn(k[head*layerHeadDim:(head+1)*layerHeadDim], kNorm, float32(cfg.RMSNormEps))
			}
		}
	}
	traceMTPVerifierLayer0Internal("q_norm", layerIdx, pos, q)
	traceMTPSummary("q_norm", traceRow, layerIdx, pos, q)
	if k != nil {
		traceMTPVerifierLayer0Internal("k_norm", layerIdx, pos, k)
		traceMTPSummary("k_norm", traceRow, layerIdx, pos, k)
		traceMTPVerifierLayer0Internal("v_norm", layerIdx, pos, v)
		traceMTPSummary("v_norm", traceRow, layerIdx, pos, v)
	}
	if cfg.ModelType == "gemma4_text" {
		freqs, rotHalf := m.ensureGemma4RoPE(layerIdx, pos)
		applyRoPEPartial(q, freqs, pos, cfg.NumHeads, layerHeadDim, rotHalf)
		if k != nil {
			applyRoPEPartial(k, freqs, pos, layerKVHeads, layerHeadDim, rotHalf)
		}
	} else {
		freqs := m.ensureRoPE(pos)
		applyRoPE(q, freqs, pos, cfg.NumHeads, layerHeadDim)
		if k != nil {
			applyRoPE(k, freqs, pos, layerKVHeads, layerHeadDim)
		}
	}
	traceMTPVerifierLayer0Internal("q_pos", layerIdx, pos, q)
	traceMTPSummary("q_pos", traceRow, layerIdx, pos, q)
	if k != nil {
		traceMTPVerifierLayer0Internal("k_pos", layerIdx, pos, k)
		traceMTPSummary("k_pos", traceRow, layerIdx, pos, k)
	}
	kvLayer := layerIdx
	if !layer.HasKV {
		kvLayer = layer.KVSourceLayer
	}
	if kvLayer < 0 || kvLayer >= len(kvCacheK) || kvLayer >= len(kvCacheV) {
		return nil, fmt.Errorf("KV layer %d out of range", kvLayer)
	}
	if k != nil {
		kvCacheK[kvLayer] = append(kvCacheK[kvLayer], k...)
		kvCacheV[kvLayer] = append(kvCacheV[kvLayer], v...)
	}
	seqLen := pos + 1
	attnSeqLen := seqLen
	attnKVOffset := 0
	if cfg.SlidingWindow > 0 && len(cfg.LayerTypes) > layerIdx && cfg.LayerTypes[layerIdx] == "sliding_attention" && seqLen > cfg.SlidingWindow {
		attnSeqLen = cfg.SlidingWindow
		attnKVOffset = seqLen - cfg.SlidingWindow
	}
	attnOut := attnOutScratch[:qDim]
	attnScores := attnScoresScratch[:attnSeqLen]
	kCache := kvCacheK[kvLayer]
	vCache := kvCacheV[kvLayer]
	if cfg.ModelType == "gemma4_text" {
		if !tryPureGoFlashAttentionInto(attnOut, q, kCache[attnKVOffset*layerKVHeads*layerHeadDim:], vCache[attnKVOffset*layerKVHeads*layerHeadDim:], layerIdx, pos, attnSeqLen, cfg.NumHeads, layerKVHeads, layerHeadDim, 1.0) {
			gqaAttentionScaleInto(attnOut, attnScores, q, kCache[attnKVOffset*layerKVHeads*layerHeadDim:], vCache[attnKVOffset*layerKVHeads*layerHeadDim:], attnSeqLen, cfg.NumHeads, layerKVHeads, layerHeadDim, 1.0)
		}
	} else {
		gqaAttentionScaleInto(attnOut, attnScores, q, kCache[attnKVOffset*layerKVHeads*layerHeadDim:], vCache[attnKVOffset*layerKVHeads*layerHeadDim:], attnSeqLen, cfg.NumHeads, layerKVHeads, layerHeadDim, float32(1.0/math.Sqrt(float64(layerHeadDim))))
	}
	traceMTPVerifierLayer0Internal("attn_pre_o", layerIdx, pos, attnOut)
	traceMTPSummary("attn_pre_o", traceRow, layerIdx, pos, attnOut)
	oOut := make([]float32, h)
	if layer.OWq != nil {
		if !m.mvQ(oOut, attnOut, layer.OWq) {
			return nil, fmt.Errorf("layer %d quantized O projection failed", layerIdx)
		}
	} else if layer.OWm != nil {
		if !mlx.GemvTo(oOut, attnOut, layer.OWm) {
			return nil, fmt.Errorf("layer %d MLX O projection failed", layerIdx)
		}
	} else if layer.OWGGUF != nil {
		if !gemvGGUFTo(oOut, attnOut, layer.OWGGUF, qDim, h) {
			return nil, fmt.Errorf("layer %d GGUF O projection failed", layerIdx)
		}
	} else {
		m.mv(oOut, attnOut, layer.OW.Data(), qDim, h)
	}
	traceMTPVerifierLayer0Internal("o_proj", layerIdx, pos, oOut)
	traceMTPSummary("o_proj", traceRow, layerIdx, pos, oOut)
	if layer.PreFFNNorm != nil {
		rmsNormInPlace(oOut, layer.PostNorm.Data(), float32(cfg.RMSNormEps))
		traceMTPSummary("attn_post_norm", traceRow, layerIdx, pos, oOut)
		simd.VecAdd(hidden, residual, oOut)
		traceMTPVerifierLayer0Internal("attn_out", layerIdx, pos, hidden)
		traceMTPSummary("attn_out", traceRow, layerIdx, pos, hidden)
		copy(residual, hidden)
	} else {
		simd.VecAdd(hidden, residual, oOut)
		traceMTPVerifierLayer0Internal("attn_out", layerIdx, pos, hidden)
		traceMTPSummary("attn_out", traceRow, layerIdx, pos, hidden)
		copy(residual, hidden)
		rmsNormInPlace(hidden, layer.PostNorm.Data(), float32(cfg.RMSNormEps))
		traceMTPSummary("attn_post_norm", traceRow, layerIdx, pos, hidden)
	}
	mlpInput := hidden
	if layer.PreFFNNorm != nil {
		mlpInput = make([]float32, h)
		copy(mlpInput, hidden)
		if cfg.ModelType == "gemma3_text" {
			simd.RMSNormBF16(mlpInput, layer.PreFFNNorm.Data(), float32(cfg.RMSNormEps))
		} else {
			rmsNormInPlace(mlpInput, layer.PreFFNNorm.Data(), float32(cfg.RMSNormEps))
		}
		traceMTPSummary("ffn_norm", traceRow, layerIdx, pos, mlpInput)
	}
	layerInter := cfg.Intermediate
	if layer.GateWq != nil && layer.GateWq.OutDim > 0 {
		layerInter = layer.GateWq.OutDim
	} else if layer.GateWm != nil && layer.GateWm.OutDim > 0 {
		layerInter = layer.GateWm.OutDim
	} else if layer.GateW != nil {
		s := layer.GateW.Shape()
		if len(s) >= 2 {
			if m.Large {
				layerInter = s[0]
			} else {
				layerInter = s[1]
			}
		} else if len(s) == 1 && s[0] > 0 {
			layerInter = s[0]
		}
	}
	var down []float32
	if layer.IsMoE && layer.ExpertGateW != nil {
		down = moeForward(mlpInput, layer, cfg)
	} else {
		gate := make([]float32, layerInter)
		up := make([]float32, layerInter)
		if layer.GateWq != nil {
			if !m.mvQ(gate, mlpInput, layer.GateWq) || !m.mvQ(up, mlpInput, layer.UpWq) {
				return nil, fmt.Errorf("layer %d quantized gate/up projection failed", layerIdx)
			}
		} else if layer.GateWm != nil {
			if !mlx.GemvTo(gate, mlpInput, layer.GateWm) {
				return nil, fmt.Errorf("layer %d MLX gate projection failed", layerIdx)
			}
			if !mlx.GemvTo(up, mlpInput, layer.UpWm) {
				return nil, fmt.Errorf("layer %d MLX up projection failed", layerIdx)
			}
		} else if layer.GateWGGUF != nil {
			if !gemvGGUFTo(gate, mlpInput, layer.GateWGGUF, h, layerInter) {
				return nil, fmt.Errorf("layer %d GGUF gate projection failed", layerIdx)
			}
			if !gemvGGUFTo(up, mlpInput, layer.UpWGGUF, h, layerInter) {
				return nil, fmt.Errorf("layer %d GGUF up projection failed", layerIdx)
			}
		} else {
			m.mv(gate, mlpInput, layer.GateW.Data(), h, layerInter)
			m.mv(up, mlpInput, layer.UpW.Data(), h, layerInter)
		}
		traceMTPSummary("ffn_gate", traceRow, layerIdx, pos, gate)
		traceMTPSummary("ffn_up", traceRow, layerIdx, pos, up)
		if cfg.ModelType == "gemma3_text" {
			simd.ToBF16(gate)
			simd.ToBF16(up)
		}
		if cfg.HiddenAct == "gelu_pytorch_tanh" {
			if cfg.ModelType == "gemma4_text" {
				ggmlGELUMulInPlace(gate, up)
			} else {
				simd.GELUTanhMul(gate, gate, up)
				if cfg.ModelType == "gemma3_text" {
					simd.ToBF16(gate)
				}
			}
		} else {
			simd.VecSiLUMul(gate, gate, up)
		}
		traceMTPSummary("ffn_geglu", traceRow, layerIdx, pos, gate)
		down = make([]float32, h)
		if layer.DownWq != nil {
			if !m.mvQ(down, gate, layer.DownWq) {
				return nil, fmt.Errorf("layer %d quantized down projection failed", layerIdx)
			}
		} else if layer.DownWm != nil {
			if !mlx.GemvTo(down, gate, layer.DownWm) {
				return nil, fmt.Errorf("layer %d MLX down projection failed", layerIdx)
			}
		} else if layer.DownWGGUF != nil {
			if !gemvGGUFTo(down, gate, layer.DownWGGUF, layerInter, h) {
				return nil, fmt.Errorf("layer %d GGUF down projection failed", layerIdx)
			}
		} else {
			m.mv(down, gate, layer.DownW.Data(), layerInter, h)
		}
		traceMTPSummary("ffn_down", traceRow, layerIdx, pos, down)
	}
	if cfg.ModelType == "gemma3_text" {
		simd.ToBF16(down)
	}
	if layer.PostFFNNorm != nil {
		if cfg.ModelType == "gemma3_text" {
			rmsNormBF16(down, layer.PostFFNNorm.Data(), float32(cfg.RMSNormEps))
		} else {
			rmsNormInPlace(down, layer.PostFFNNorm.Data(), float32(cfg.RMSNormEps))
		}
		traceMTPSummary("ffn_post_norm", traceRow, layerIdx, pos, down)
	}
	simd.VecAdd(hidden, residual, down)
	traceMTPVerifierLayer0Internal("ffn_resid", layerIdx, pos, hidden)
	traceMTPSummary("ffn_resid", traceRow, layerIdx, pos, hidden)
	if (layer.PLIGate != nil || layer.PLIGateGGUF != nil) && perLayerInputs != nil && layerIdx < len(perLayerInputs) {
		traceMTPSummary("pe_in", traceRow, layerIdx, pos, hidden)
		hpl := cfg.HiddenPerLayer
		pli := perLayerInputs[layerIdx]
		gate2 := make([]float32, hpl)
		if layer.PLIGateGGUF != nil {
			if !gemvGGUFTo(gate2, hidden, layer.PLIGateGGUF, h, hpl) {
				return nil, fmt.Errorf("layer %d GGUF PLI gate projection failed", layerIdx)
			}
		} else {
			gemvNT(gate2, hidden, layer.PLIGate, h, hpl)
		}
		traceMTPSummary("per_layer_gate", traceRow, layerIdx, pos, gate2)
		if cfg.ModelType == "gemma4_text" {
			for i := range gate2 {
				gate2[i] = ggmlGELUF32(gate2[i])
			}
			traceMTPSummary("per_layer_gate_gelu", traceRow, layerIdx, pos, gate2)
			pliMul := pli
			if mtpRoundPLIRowEnabled() {
				pliMul = make([]float32, len(pli))
				for i, v := range pli {
					pliMul[i] = half.F16ToF32(half.F32ToF16(v))
				}
			}
			simd.VecMul(gate2, gate2, pliMul)
		} else {
			simd.GELUTanhMul(gate2, gate2, pli)
			traceMTPSummary("per_layer_gate_gelu", traceRow, layerIdx, pos, gate2)
		}
		proj2 := make([]float32, h)
		if layer.PLIProjGGUF != nil {
			if !gemvGGUFTo(proj2, gate2, layer.PLIProjGGUF, hpl, h) {
				return nil, fmt.Errorf("layer %d GGUF PLI projection failed", layerIdx)
			}
		} else {
			gemvNT(proj2, gate2, layer.PLIProj, hpl, h)
		}
		rmsNormInPlace(proj2, layer.PLIPostNorm, float32(cfg.RMSNormEps))
		traceMTPVerifierLayer0Internal("pli_out", layerIdx, pos, proj2)
		traceMTPSummary("per_layer_embd_out", traceRow, layerIdx, pos, proj2)
		simd.VecAdd(hidden, hidden, proj2)
	}
	if layer.LayerScalar != 1.0 {
		simd.VecScale(hidden, hidden, layer.LayerScalar)
	}
	traceMTPSummary("l_out_pre_bf16", traceRow, layerIdx, pos, hidden)
	if cfg.ModelType == "gemma3_text" && !mtpSkipLayerBF16(layerIdx, pos) {
		simd.ToBF16(hidden)
	}
	traceMTPVerifierLayer0Internal("l0_out", layerIdx, pos, hidden)
	traceMTPSummary("l_out", traceRow, layerIdx, pos, hidden)
	return hidden, nil
}
