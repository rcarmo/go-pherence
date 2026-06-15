package model

import "fmt"

// MTPVerifierBatchInputs is the explicit input bundle for the verifier graph:
// token embeddings plus optional Gemma4 per-layer inputs for every verifier row
// in [input]+drafted order. Today RunMTPVerifierForward consumes this bundle
// sequentially; the structure is the handoff point for a future batched verifier
// runner that matches llama.cpp/LiteRT verify graph construction.
type MTPVerifierBatchInputs struct {
	Plan              MTPVerifierPlan
	HiddenRows        [][]float32
	PerLayerInputs    [][][]float32
	HasPerLayerInputs bool
	Attention         MTPVerifierAttentionPlan
	Scratch           MTPVerifierBatchScratchPlan
}

func NewMTPVerifierBatchInputs(m *LlamaModel, plan MTPVerifierPlan) (MTPVerifierBatchInputs, error) {
	if err := validateMTPVerifierPlanForModel(m, plan); err != nil {
		return MTPVerifierBatchInputs{}, err
	}
	rows := make([][]float32, len(plan.VerifierTokens))
	pliRows := make([][][]float32, len(plan.VerifierTokens))
	hasPLI := false
	for i, tok := range plan.VerifierTokens {
		hidden := make([]float32, m.Config.HiddenSize)
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
			pliRows[i] = clonePerLayerInputs(pli)
		}
	}
	attn, err := NewMTPVerifierAttentionPlan(m, plan)
	if err != nil {
		return MTPVerifierBatchInputs{}, err
	}
	out := MTPVerifierBatchInputs{Plan: cloneMTPVerifierPlanValue(plan), HiddenRows: rows, PerLayerInputs: pliRows, HasPerLayerInputs: hasPLI, Attention: attn}
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
