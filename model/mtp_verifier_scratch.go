package model

import "fmt"

// MTPVerifierBatchScratchPlan is a backend-neutral allocation contract for the
// verifier batch. It records per-layer buffer widths and conservative total
// scratch needs so CPU/SIMD/GPU lowering paths can share the same shape checks.
type MTPVerifierBatchScratchPlan struct {
	Batch            int
	HiddenSize       int
	MaxQDim          int
	MaxKVDim         int
	MaxIntermediate  int
	MaxAttentionRows int
	Layers           []MTPVerifierLayerScratchPlan
	TotalFloat32     int
}

type MTPVerifierLayerScratchPlan struct {
	Layer                int
	Batch                int
	HiddenSize           int
	QDim                 int
	KVDim                int
	Intermediate         int
	HasKV                bool
	SharedKV             bool
	AttentionRows        int
	AttentionOutFloats   int
	AttentionScoreFloats int
	MLPFloats            int
	PLIFloats            int
	TotalFloat32         int
}

func NewMTPVerifierBatchScratchPlan(m *LlamaModel, batch MTPVerifierBatchInputs) (MTPVerifierBatchScratchPlan, error) {
	if err := validateMTPVerifierPlanForModel(m, batch.Plan); err != nil {
		return MTPVerifierBatchScratchPlan{}, err
	}
	B := len(batch.Plan.VerifierTokens)
	if B <= 0 || len(batch.HiddenRows) != B {
		return MTPVerifierBatchScratchPlan{}, fmt.Errorf("invalid verifier batch rows=%d tokens=%d", len(batch.HiddenRows), B)
	}
	h := m.Config.HiddenSize
	out := MTPVerifierBatchScratchPlan{Batch: B, HiddenSize: h, Layers: make([]MTPVerifierLayerScratchPlan, m.Config.NumLayers)}
	for l := 0; l < m.Config.NumLayers; l++ {
		layer := &m.Layers[l]
		headDim, err := m.LayerHeadDim(l)
		if err != nil {
			return MTPVerifierBatchScratchPlan{}, err
		}
		qDim, ok := checkedProduct(m.Config.NumHeads, headDim)
		if headDim <= 0 || !ok {
			return MTPVerifierBatchScratchPlan{}, fmt.Errorf("invalid verifier scratch attention dims layer=%d heads=%d headDim=%d", l, m.Config.NumHeads, headDim)
		}
		kvDim, err := m.LayerKVDim(l)
		if err != nil {
			return MTPVerifierBatchScratchPlan{}, err
		}
		if kvDim == 0 && !layer.HasKV {
			if layer.KVSourceLayer < 0 || layer.KVSourceLayer >= m.Config.NumLayers {
				return MTPVerifierBatchScratchPlan{}, fmt.Errorf("shared-KV verifier scratch layer %d source %d out of range", l, layer.KVSourceLayer)
			}
			kvDim, err = m.LayerKVDim(layer.KVSourceLayer)
			if err != nil {
				return MTPVerifierBatchScratchPlan{}, err
			}
			if kvDim == 0 {
				return MTPVerifierBatchScratchPlan{}, fmt.Errorf("shared-KV verifier scratch layer %d source %d does not append KV", l, layer.KVSourceLayer)
			}
		}
		inter := m.layerInterFor(layer)
		if inter < 0 {
			return MTPVerifierBatchScratchPlan{}, fmt.Errorf("invalid verifier scratch intermediate layer=%d intermediate=%d", l, inter)
		}
		attnRows := 0
		if l < len(batch.Attention.Layers) && len(batch.Attention.Layers[l].KVEndExclusive) > 0 {
			for _, end := range batch.Attention.Layers[l].KVEndExclusive {
				if end > attnRows {
					attnRows = end
				}
			}
		}
		attnOut, okAttnOut := checkedProduct(B, qDim)
		attnScores, okScores := checkedProduct(B, attnRows)
		mlp, okMLP := checkedProduct(B, inter)
		pli := 0
		if batch.HasPerLayerInputs && m.Config.HiddenPerLayer > 0 {
			var okPLI bool
			pli, okPLI = checkedProduct(B, m.Config.HiddenPerLayer)
			if !okPLI {
				return MTPVerifierBatchScratchPlan{}, fmt.Errorf("verifier scratch PLI size overflow layer=%d", l)
			}
		}
		if !okAttnOut || !okScores || !okMLP {
			return MTPVerifierBatchScratchPlan{}, fmt.Errorf("verifier scratch size overflow layer=%d", l)
		}
		total, okTotal := checkedAddNonNegative(attnOut, attnScores)
		if okTotal {
			total, okTotal = checkedAddNonNegative(total, mlp)
		}
		if okTotal {
			total, okTotal = checkedAddNonNegative(total, pli)
		}
		if !okTotal {
			return MTPVerifierBatchScratchPlan{}, fmt.Errorf("verifier scratch total size overflow layer=%d", l)
		}
		lp := MTPVerifierLayerScratchPlan{
			Layer: l, Batch: B, HiddenSize: h, QDim: qDim, KVDim: kvDim,
			Intermediate: inter, HasKV: layer.HasKV, SharedKV: !layer.HasKV,
			AttentionRows: attnRows, AttentionOutFloats: attnOut, AttentionScoreFloats: attnScores,
			MLPFloats: mlp, PLIFloats: pli, TotalFloat32: total,
		}
		out.Layers[l] = lp
		if qDim > out.MaxQDim {
			out.MaxQDim = qDim
		}
		if kvDim > out.MaxKVDim {
			out.MaxKVDim = kvDim
		}
		if inter > out.MaxIntermediate {
			out.MaxIntermediate = inter
		}
		if attnRows > out.MaxAttentionRows {
			out.MaxAttentionRows = attnRows
		}
		var okPlan bool
		out.TotalFloat32, okPlan = checkedAddNonNegative(out.TotalFloat32, total)
		if !okPlan {
			return MTPVerifierBatchScratchPlan{}, fmt.Errorf("verifier scratch plan total size overflow")
		}
	}
	return out, nil
}
