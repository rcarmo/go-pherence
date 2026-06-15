package model

import (
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	gemmacfg "github.com/rcarmo/go-pherence/model/gemma"
)

type MTPVerifierLayerQKVBatch struct {
	Q        []float32
	K        []float32
	V        []float32
	QDim     int
	KVDim    int
	HeadDim  int
	KVHeads  int
	HasKV    bool
	NormedIn []float32
}

// ProjectMTPVerifierLayerQKVBatch computes the verifier layer's input norm and
// Q/K/V projections for all verifier rows in one batch contract. Dense/MLX
// weights use batched projection helpers; quantized QAT weights currently keep
// the exact per-row m.mvQ path as the SIMD oracle until a true quantized batch
// kernel is introduced.
func (m *LlamaModel) ProjectMTPVerifierLayerQKVBatch(batch MTPVerifierBatchInputs, layerIdx int, hiddenFlat []float32) (MTPVerifierLayerQKVBatch, error) {
	if m == nil {
		return MTPVerifierLayerQKVBatch{}, fmt.Errorf("nil model")
	}
	if layerIdx < 0 || layerIdx >= m.Config.NumLayers || layerIdx >= len(m.Layers) {
		return MTPVerifierLayerQKVBatch{}, fmt.Errorf("layer index %d out of range", layerIdx)
	}
	if err := validateMTPVerifierPlanForModel(m, batch.Plan); err != nil {
		return MTPVerifierLayerQKVBatch{}, err
	}
	B := len(batch.Plan.VerifierTokens)
	h := m.Config.HiddenSize
	if B <= 0 || h <= 0 || len(hiddenFlat) < B*h {
		return MTPVerifierLayerQKVBatch{}, fmt.Errorf("invalid verifier QKV batch hidden len=%d B=%d hidden=%d", len(hiddenFlat), B, h)
	}
	layer := &m.Layers[layerIdx]
	if layer.InputNorm == nil {
		return MTPVerifierLayerQKVBatch{}, fmt.Errorf("layer %d missing input norm", layerIdx)
	}
	headDim := m.Config.HeadDim
	if layer.HeadDimLocal > 0 {
		headDim = layer.HeadDimLocal
	}
	kvHeads := gemmacfg.LayerKVHeads(m.Config, layerIdx)
	qDim, okQ := checkedProduct(m.Config.NumHeads, headDim)
	kvDim, okKV := checkedProduct(kvHeads, headDim)
	if headDim <= 0 || kvHeads < 0 || !okQ || !okKV {
		return MTPVerifierLayerQKVBatch{}, fmt.Errorf("invalid verifier QKV dims layer=%d heads=%d kvHeads=%d headDim=%d", layerIdx, m.Config.NumHeads, kvHeads, headDim)
	}
	normed := make([]float32, B*h)
	copy(normed, hiddenFlat[:B*h])
	isGemma := m.Config.ModelType == "gemma3_text" || m.Config.ModelType == "gemma4_text"
	for b := 0; b < B; b++ {
		row := normed[b*h : (b+1)*h]
		if isGemma {
			simd.RMSNormBF16(row, layer.InputNorm.Data(), float32(m.Config.RMSNormEps))
		} else {
			rmsNormInPlace(row, layer.InputNorm.Data(), float32(m.Config.RMSNormEps))
		}
	}
	q := make([]float32, B*qDim)
	if layer.QWq != nil {
		for b := 0; b < B; b++ {
			m.mvQ(q[b*qDim:(b+1)*qDim], normed[b*h:(b+1)*h], layer.QWq)
		}
	} else if !m.projBatch(q, normed, B, layer.QW, layer.QWm, h, qDim) {
		return MTPVerifierLayerQKVBatch{}, fmt.Errorf("layer %d Q batch projection rejected", layerIdx)
	}
	var k, v []float32
	if layer.HasKV {
		k = make([]float32, B*kvDim)
		v = make([]float32, B*kvDim)
		if layer.KWq != nil {
			for b := 0; b < B; b++ {
				m.mvQ(k[b*kvDim:(b+1)*kvDim], normed[b*h:(b+1)*h], layer.KWq)
				if m.Config.AttentionKEqV && (layer.VWq == nil || layer.VWq == layer.KWq) {
					copy(v[b*kvDim:(b+1)*kvDim], k[b*kvDim:(b+1)*kvDim])
				} else if layer.VWq != nil {
					m.mvQ(v[b*kvDim:(b+1)*kvDim], normed[b*h:(b+1)*h], layer.VWq)
				} else {
					return MTPVerifierLayerQKVBatch{}, fmt.Errorf("layer %d missing quantized V projection", layerIdx)
				}
			}
		} else if layer.KWm != nil {
			if !m.projBatch(k, normed, B, layer.KW, layer.KWm, h, kvDim) {
				return MTPVerifierLayerQKVBatch{}, fmt.Errorf("layer %d K batch projection rejected", layerIdx)
			}
			if m.Config.AttentionKEqV && (layer.VWm == nil || layer.VWm == layer.KWm) {
				copy(v, k)
			} else if !m.projBatch(v, normed, B, layer.VW, layer.VWm, h, kvDim) {
				return MTPVerifierLayerQKVBatch{}, fmt.Errorf("layer %d V batch projection rejected", layerIdx)
			}
		} else {
			if !m.projBatch(k, normed, B, layer.KW, nil, h, kvDim) {
				return MTPVerifierLayerQKVBatch{}, fmt.Errorf("layer %d K batch projection rejected", layerIdx)
			}
			if m.Config.AttentionKEqV && (layer.VW == nil || layer.VW == layer.KW) {
				copy(v, k)
			} else if !m.projBatch(v, normed, B, layer.VW, nil, h, kvDim) {
				return MTPVerifierLayerQKVBatch{}, fmt.Errorf("layer %d V batch projection rejected", layerIdx)
			}
		}
	}
	for b := 0; b < B; b++ {
		qRow := q[b*qDim : (b+1)*qDim]
		var kRow, vRow []float32
		if k != nil {
			kRow = k[b*kvDim : (b+1)*kvDim]
			vRow = v[b*kvDim : (b+1)*kvDim]
		}
		postProcessMTPVerifierQKV(m, layer, layerIdx, qRow, kRow, vRow, batch.Plan.Positions[b], headDim, kvHeads)
	}
	return MTPVerifierLayerQKVBatch{Q: q, K: k, V: v, QDim: qDim, KVDim: kvDim, HeadDim: headDim, KVHeads: kvHeads, HasKV: layer.HasKV, NormedIn: normed}, nil
}

func postProcessMTPVerifierQKV(m *LlamaModel, layer *LlamaLayer, layerIdx int, q, k, v []float32, pos, headDim, kvHeads int) {
	cfg := m.Config
	isGemma := cfg.ModelType == "gemma3_text" || cfg.ModelType == "gemma4_text"
	if isGemma {
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
	if isGemma {
		normFn = rmsNormBF16
	}
	if cfg.ModelType == "gemma4_text" && v != nil {
		eps := float32(cfg.RMSNormEps)
		for head := 0; head < kvHeads; head++ {
			simd.RMSNormNoScale(v[head*headDim:(head+1)*headDim], eps)
		}
	} else if layer.VNorm != nil && v != nil {
		vnorm := layer.VNorm.Data()
		for head := 0; head < kvHeads; head++ {
			normFn(v[head*headDim:(head+1)*headDim], vnorm, float32(cfg.RMSNormEps))
		}
	}
	if layer.QNorm != nil {
		qNorm := layer.QNorm.Data()
		for head := 0; head < cfg.NumHeads; head++ {
			normFn(q[head*headDim:(head+1)*headDim], qNorm, float32(cfg.RMSNormEps))
		}
		if k != nil && layer.KNorm != nil {
			kNorm := layer.KNorm.Data()
			for head := 0; head < kvHeads; head++ {
				normFn(k[head*headDim:(head+1)*headDim], kNorm, float32(cfg.RMSNormEps))
			}
		}
	}
	if cfg.ModelType == "gemma4_text" && m.RopeFreqsSWA != nil {
		isSWA := true
		if len(cfg.LayerTypes) > layerIdx {
			isSWA = cfg.LayerTypes[layerIdx] == "sliding_attention"
		}
		if isSWA {
			applyRoPEPartial(q, m.RopeFreqsSWA, pos, cfg.NumHeads, headDim, m.RopeHalfSWA)
			if k != nil {
				applyRoPEPartial(k, m.RopeFreqsSWA, pos, kvHeads, headDim, m.RopeHalfSWA)
			}
		} else {
			applyRoPEPartial(q, m.RopeFreqsFull, pos, cfg.NumHeads, headDim, m.RopeHalfFull)
			if k != nil {
				applyRoPEPartial(k, m.RopeFreqsFull, pos, kvHeads, headDim, m.RopeHalfFull)
			}
		}
	} else {
		applyRoPE(q, m.RopeFreqs, pos, cfg.NumHeads, headDim)
		if k != nil {
			applyRoPE(k, m.RopeFreqs, pos, kvHeads, headDim)
		}
	}
}
