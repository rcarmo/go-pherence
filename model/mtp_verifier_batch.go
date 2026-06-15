package model

import "fmt"

// MTPVerifierBatchInputs is the explicit input bundle for the verifier graph:
// token embeddings plus optional Gemma4 per-layer inputs for every verifier row
// in [input]+drafted order. Today RunMTPVerifierForward consumes this bundle
// sequentially; the structure is the handoff point for a future batched verifier
// runner that matches llama.cpp/LiteRT verify graph construction.
type MTPVerifierBatchInputs struct {
	Plan              MTPVerifierPlan
	HiddenFlat        []float32
	HiddenRows        [][]float32
	PerLayerInputFlat []float32
	PerLayerInputs    [][][]float32
	HasPerLayerInputs bool
	Attention         MTPVerifierAttentionPlan
	Scratch           MTPVerifierBatchScratchPlan
}

func NewMTPVerifierBatchInputs(m *LlamaModel, plan MTPVerifierPlan) (MTPVerifierBatchInputs, error) {
	if err := validateMTPVerifierPlanForModel(m, plan); err != nil {
		return MTPVerifierBatchInputs{}, err
	}
	B := len(plan.VerifierTokens)
	h := m.Config.HiddenSize
	hiddenN, okHidden := checkedProduct(B, h)
	if !okHidden {
		return MTPVerifierBatchInputs{}, fmt.Errorf("verifier hidden batch dimension overflow B=%d hidden=%d", B, h)
	}
	rows := make([][]float32, B)
	hiddenFlat := make([]float32, hiddenN)
	pliRows := make([][][]float32, B)
	hasPLI := false
	var pliFlat []float32
	for i, tok := range plan.VerifierTokens {
		hidden := hiddenFlat[i*h : (i+1)*h]
		if err := m.ScaledTokenEmbeddingInto(hidden, tok); err != nil {
			return MTPVerifierBatchInputs{}, fmt.Errorf("verifier token %d embedding: %w", i, err)
		}
		rows[i] = hidden
		pli, err := m.Gemma4PerLayerInputs(hidden, tok)
		if err != nil {
			return MTPVerifierBatchInputs{}, fmt.Errorf("verifier token %d per-layer inputs: %w", i, err)
		}
		if pli != nil {
			hasPLI = true
			perTokenPLI, okPerToken := checkedProduct(m.Config.NumLayers, m.Config.HiddenPerLayer)
			if !okPerToken {
				return MTPVerifierBatchInputs{}, fmt.Errorf("verifier PLI per-token dimension overflow")
			}
			if pliFlat == nil {
				totalPLI, ok := checkedProduct(B, perTokenPLI)
				if !ok {
					return MTPVerifierBatchInputs{}, fmt.Errorf("verifier PLI batch dimension overflow")
				}
				pliFlat = make([]float32, totalPLI)
			}
			pliRows[i] = make([][]float32, len(pli))
			rowBase, okRowBase := checkedProduct(i, perTokenPLI)
			if !okRowBase {
				return MTPVerifierBatchInputs{}, fmt.Errorf("verifier PLI row offset overflow token=%d perToken=%d", i, perTokenPLI)
			}
			for l := range pli {
				layerOff, okLayerOff := checkedProduct(l, m.Config.HiddenPerLayer)
				if !okLayerOff {
					return MTPVerifierBatchInputs{}, fmt.Errorf("verifier PLI layer offset overflow layer=%d hiddenPerLayer=%d", l, m.Config.HiddenPerLayer)
				}
				start := rowBase + layerOff
				end := start + m.Config.HiddenPerLayer
				if start < rowBase || end < start || end > len(pliFlat) {
					return MTPVerifierBatchInputs{}, fmt.Errorf("verifier PLI flat range [%d,%d) outside len=%d", start, end, len(pliFlat))
				}
				dst := pliFlat[start:end]
				copy(dst, pli[l])
				pliRows[i][l] = dst
			}
		}
	}
	attn, err := NewMTPVerifierAttentionPlan(m, plan)
	if err != nil {
		return MTPVerifierBatchInputs{}, err
	}
	out := MTPVerifierBatchInputs{Plan: cloneMTPVerifierPlanValue(plan), HiddenFlat: hiddenFlat, HiddenRows: rows, PerLayerInputFlat: pliFlat, PerLayerInputs: pliRows, HasPerLayerInputs: hasPLI, Attention: attn}
	scratch, err := NewMTPVerifierBatchScratchPlan(m, out)
	if err != nil {
		return MTPVerifierBatchInputs{}, err
	}
	out.Scratch = scratch
	return out, nil
}

func validateMTPVerifierPlanForModel(m *LlamaModel, plan MTPVerifierPlan) error {
	if m == nil {
		return fmt.Errorf("nil model")
	}
	if len(plan.VerifierTokens) == 0 {
		return fmt.Errorf("empty verifier plan")
	}
	if len(plan.VerifierTokens) != len(plan.Positions) {
		return fmt.Errorf("verifier plan tokens=%d positions=%d", len(plan.VerifierTokens), len(plan.Positions))
	}
	if plan.InputToken != plan.VerifierTokens[0] {
		return fmt.Errorf("verifier plan input token=%d does not match first verifier token=%d", plan.InputToken, plan.VerifierTokens[0])
	}
	if len(plan.DraftedTokens)+1 != len(plan.VerifierTokens) {
		return fmt.Errorf("verifier plan drafted=%d tokens=%d", len(plan.DraftedTokens), len(plan.VerifierTokens))
	}
	vocab := m.Config.VocabSize
	if vocab <= 0 || m.Config.HiddenSize <= 0 || m.Config.NumLayers < 0 || len(m.Layers) < m.Config.NumLayers {
		return fmt.Errorf("invalid verifier model dims vocab=%d hidden=%d layers=%d/%d", vocab, m.Config.HiddenSize, m.Config.NumLayers, len(m.Layers))
	}
	for i, tok := range plan.VerifierTokens {
		if tok < 0 || tok >= vocab {
			return fmt.Errorf("verifier token %d at index %d out of range [0,%d)", tok, i, vocab)
		}
	}
	for i, tok := range plan.DraftedTokens {
		if tok != plan.VerifierTokens[i+1] {
			return fmt.Errorf("drafted token %d=%d does not match verifier token %d", i, tok, plan.VerifierTokens[i+1])
		}
	}
	wantPositions, err := mtpVerifierPositions(plan.StartPos, len(plan.VerifierTokens))
	if err != nil {
		return err
	}
	for i, pos := range plan.Positions {
		if pos != wantPositions[i] {
			return fmt.Errorf("verifier plan position %d=%d, want %d", i, pos, wantPositions[i])
		}
	}
	return nil
}

func clonePerLayerInputs(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}

func cloneMTPVerifierPlanValue(plan MTPVerifierPlan) MTPVerifierPlan {
	return MTPVerifierPlan{
		InputToken:     plan.InputToken,
		DraftedTokens:  append([]int(nil), plan.DraftedTokens...),
		VerifierTokens: append([]int(nil), plan.VerifierTokens...),
		StartPos:       plan.StartPos,
		Positions:      append([]int(nil), plan.Positions...),
	}
}
