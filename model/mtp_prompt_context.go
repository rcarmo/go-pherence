package model

import (
	"fmt"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
	"math"

	"github.com/rcarmo/go-pherence/backends/mlx"
	"github.com/rcarmo/go-pherence/backends/simd/runtime"
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

// BuildMTPPromptContext runs the prompt through the same CPU token/layer path
// used by Generate, including Gemma4 per-layer inputs, and returns copied final
// activation plus float KV caches. It is intended for MTP wiring/smokes, not as
// a high-throughput prefill implementation.
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
		if m.Layers[i].HeadDimLocal > maxHeadDim {
			maxHeadDim = m.Layers[i].HeadDimLocal
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
	cfg := m.Config
	h := cfg.HiddenSize
	layer := &m.Layers[layerIdx]
	if len(hidden) < h || layer.InputNorm == nil || layer.PostNorm == nil {
		return nil, fmt.Errorf("invalid hidden/norms")
	}
	residual := make([]float32, h)
	copy(residual, hidden[:h])
	layerHeadDim := cfg.HeadDim
	if layer.HeadDimLocal > 0 {
		layerHeadDim = layer.HeadDimLocal
	}
	layerKVHeads := gemmacfg.LayerKVHeads(cfg, layerIdx)
	qDim, ok := checkedProduct(cfg.NumHeads, layerHeadDim)
	layerKVDim, okKV := checkedProduct(layerKVHeads, layerHeadDim)
	if layerHeadDim <= 0 || !ok || !okKV {
		return nil, fmt.Errorf("invalid attention dims")
	}
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		simd.RMSNormBF16(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))
	} else {
		rmsNormInPlace(hidden, layer.InputNorm.Data(), float32(cfg.RMSNormEps))
	}
	q := make([]float32, qDim)
	if layer.QWq != nil {
		m.mvQ(q, hidden, layer.QWq)
	} else if layer.QWm != nil {
		mlx.Gemv(q, hidden, layer.QWm)
	} else {
		m.mv(q, hidden, layer.QW.Data(), h, qDim)
	}
	var k, v []float32
	if layer.HasKV {
		k = make([]float32, layerKVDim)
		v = make([]float32, layerKVDim)
		if layer.KWq != nil {
			m.mvQ(k, hidden, layer.KWq)
			m.mvQ(v, hidden, layer.VWq)
		} else if layer.KWm != nil {
			mlx.Gemv(k, hidden, layer.KWm)
			if layer.VWm == layer.KWm && cfg.AttentionKEqV {
				copy(v, k)
			} else {
				mlx.Gemv(v, hidden, layer.VWm)
			}
		} else {
			m.mv(k, hidden, layer.KW.Data(), h, layerKVDim)
			if layer.VW == layer.KW && cfg.AttentionKEqV {
				copy(v, k)
			} else {
				m.mv(v, hidden, layer.VW.Data(), h, layerKVDim)
			}
		}
	}
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
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
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
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
	if cfg.ModelType == "gemma4_text" && m.RopeFreqsSWA != nil {
		isSWA := true
		if len(cfg.LayerTypes) > layerIdx {
			isSWA = cfg.LayerTypes[layerIdx] == "sliding_attention"
		}
		if isSWA {
			applyRoPEPartial(q, m.RopeFreqsSWA, pos, cfg.NumHeads, layerHeadDim, m.RopeHalfSWA)
			if k != nil {
				applyRoPEPartial(k, m.RopeFreqsSWA, pos, layerKVHeads, layerHeadDim, m.RopeHalfSWA)
			}
		} else {
			applyRoPEPartial(q, m.RopeFreqsFull, pos, cfg.NumHeads, layerHeadDim, m.RopeHalfFull)
			if k != nil {
				applyRoPEPartial(k, m.RopeFreqsFull, pos, layerKVHeads, layerHeadDim, m.RopeHalfFull)
			}
		}
	} else {
		applyRoPE(q, m.RopeFreqs, pos, cfg.NumHeads, layerHeadDim)
		if k != nil {
			applyRoPE(k, m.RopeFreqs, pos, layerKVHeads, layerHeadDim)
		}
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
		gqaAttentionScaleInto(attnOut, attnScores, q, kCache[attnKVOffset*layerKVHeads*layerHeadDim:], vCache[attnKVOffset*layerKVHeads*layerHeadDim:], attnSeqLen, cfg.NumHeads, layerKVHeads, layerHeadDim, 1.0)
	} else {
		gqaAttentionScaleInto(attnOut, attnScores, q, kCache[attnKVOffset*layerKVHeads*layerHeadDim:], vCache[attnKVOffset*layerKVHeads*layerHeadDim:], attnSeqLen, cfg.NumHeads, layerKVHeads, layerHeadDim, float32(1.0/math.Sqrt(float64(layerHeadDim))))
	}
	oOut := make([]float32, h)
	if layer.OWq != nil {
		m.mvQ(oOut, attnOut, layer.OWq)
	} else if layer.OWm != nil {
		mlx.Gemv(oOut, attnOut, layer.OWm)
	} else {
		m.mv(oOut, attnOut, layer.OW.Data(), qDim, h)
	}
	if layer.PreFFNNorm != nil {
		rmsNormInPlace(oOut, layer.PostNorm.Data(), float32(cfg.RMSNormEps))
		simd.VecAdd(hidden, residual, oOut)
		copy(residual, hidden)
	} else {
		simd.VecAdd(hidden, residual, oOut)
		copy(residual, hidden)
		rmsNormInPlace(hidden, layer.PostNorm.Data(), float32(cfg.RMSNormEps))
	}
	mlpInput := hidden
	if layer.PreFFNNorm != nil {
		mlpInput = make([]float32, h)
		copy(mlpInput, hidden)
		if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
			simd.RMSNormBF16(mlpInput, layer.PreFFNNorm.Data(), float32(cfg.RMSNormEps))
		} else {
			rmsNormInPlace(mlpInput, layer.PreFFNNorm.Data(), float32(cfg.RMSNormEps))
		}
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
			m.mvQ(gate, mlpInput, layer.GateWq)
			m.mvQ(up, mlpInput, layer.UpWq)
		} else if layer.GateWm != nil {
			mlx.Gemv(gate, mlpInput, layer.GateWm)
			mlx.Gemv(up, mlpInput, layer.UpWm)
		} else {
			m.mv(gate, mlpInput, layer.GateW.Data(), h, layerInter)
			m.mv(up, mlpInput, layer.UpW.Data(), h, layerInter)
		}
		if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
			simd.ToBF16(gate)
			simd.ToBF16(up)
		}
		if cfg.HiddenAct == "gelu_pytorch_tanh" {
			simd.GELUTanhMul(gate, gate, up)
			simd.ToBF16(gate)
		} else {
			simd.VecSiLUMul(gate, gate, up)
		}
		down = make([]float32, h)
		if layer.DownWq != nil {
			m.mvQ(down, gate, layer.DownWq)
		} else if layer.DownWm != nil {
			mlx.Gemv(down, gate, layer.DownWm)
		} else {
			m.mv(down, gate, layer.DownW.Data(), layerInter, h)
		}
	}
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		simd.ToBF16(down)
	}
	if layer.PostFFNNorm != nil {
		if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
			rmsNormBF16(down, layer.PostFFNNorm.Data(), float32(cfg.RMSNormEps))
		} else {
			rmsNormInPlace(down, layer.PostFFNNorm.Data(), float32(cfg.RMSNormEps))
		}
	}
	simd.VecAdd(hidden, residual, down)
	if layer.PLIGate != nil && perLayerInputs != nil && layerIdx < len(perLayerInputs) {
		hpl := cfg.HiddenPerLayer
		pli := perLayerInputs[layerIdx]
		gate2 := make([]float32, hpl)
		gemvNT(gate2, hidden, layer.PLIGate, h, hpl)
		simd.GELUTanhMul(gate2, gate2, pli)
		proj2 := make([]float32, h)
		gemvNT(proj2, gate2, layer.PLIProj, hpl, h)
		rmsNormInPlace(proj2, layer.PLIPostNorm, float32(cfg.RMSNormEps))
		simd.VecAdd(hidden, hidden, proj2)
	}
	if layer.LayerScalar != 1.0 {
		simd.VecScale(hidden, hidden, layer.LayerScalar)
	}
	if cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text" {
		simd.ToBF16(hidden)
	}
	return hidden, nil
}
