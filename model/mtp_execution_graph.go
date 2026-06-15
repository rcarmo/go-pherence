package model

import "fmt"

// MTPExecutionGraph is the explicit per-iteration graph contract for Gemma4
// MTP/speculative decoding. It mirrors the llama.cpp/LiteRT shape: run G
// hidden-state-conditioned drafter steps, verify [input]+drafted in one main
// model pass, then keep accepted-prefix plus verifier bonus KV.
type MTPExecutionGraph struct {
	InputToken      int
	DraftedTokens   []int
	StartPos        int
	DrafterSteps    []MTPDrafterGraphStep
	Verifier        MTPVerifierPlan
	MaxKVKeepTokens int
}

type MTPDrafterGraphStep struct {
	Index            int
	InputToken       int
	ActivationWidth  int
	ExternalKVSeqLen int
	ExternalKVLayers []int
}

type MTPKVCommitPlan struct {
	KeepTokens   int
	Positions    []int
	OutputTokens []int
}

func NewMTPExecutionGraph(m *LlamaModel, d *Gemma4MTPDrafter, state MTPDrafterState, externalKV *MTPDrafterExternalKV, drafted []int, startPos int) (MTPExecutionGraph, error) {
	if m == nil {
		return MTPExecutionGraph{}, fmt.Errorf("nil model")
	}
	if d == nil {
		return MTPExecutionGraph{}, fmt.Errorf("nil drafter")
	}
	if len(drafted) > maxMTPDraftCount {
		return MTPExecutionGraph{}, fmt.Errorf("draft count %d out of range [0,%d]", len(drafted), maxMTPDraftCount)
	}
	if state.PreviousToken < 0 || state.PreviousToken >= m.Config.VocabSize {
		return MTPExecutionGraph{}, fmt.Errorf("previous token %d out of verifier vocab [0,%d)", state.PreviousToken, m.Config.VocabSize)
	}
	if d.BackboneHiddenSize <= 0 || len(state.Activation) != d.BackboneHiddenSize {
		return MTPExecutionGraph{}, fmt.Errorf("drafter activation width=%d want backbone=%d", len(state.Activation), d.BackboneHiddenSize)
	}
	if externalKV != nil {
		if err := validateMTPDrafterExternalKV(d, externalKV); err != nil {
			return MTPExecutionGraph{}, err
		}
	}
	verifier, err := NewMTPVerifierPlan(m, state.PreviousToken, drafted, startPos)
	if err != nil {
		return MTPExecutionGraph{}, err
	}
	steps := make([]MTPDrafterGraphStep, len(drafted))
	extSeqLen := 0
	var extLayers []int
	if externalKV != nil {
		extSeqLen = externalKV.SeqLen
		extLayers = append([]int(nil), externalKV.SourceLayers...)
	}
	for i := range drafted {
		input := state.PreviousToken
		if i > 0 {
			input = drafted[i-1]
		}
		steps[i] = MTPDrafterGraphStep{
			Index:            i,
			InputToken:       input,
			ActivationWidth:  d.BackboneHiddenSize,
			ExternalKVSeqLen: extSeqLen,
			ExternalKVLayers: append([]int(nil), extLayers...),
		}
	}
	return MTPExecutionGraph{
		InputToken:      state.PreviousToken,
		DraftedTokens:   append([]int(nil), drafted...),
		StartPos:        startPos,
		DrafterSteps:    steps,
		Verifier:        verifier,
		MaxKVKeepTokens: len(drafted) + 1,
	}, nil
}

// CommitPlan returns the exact verifier KV window to retain after acceptance:
// accepted draft prefix plus the verifier bonus token. The returned positions
// are a prefix of the verifier pass positions and are safe to pass to KV commit
// helpers that retain appended positions.
func (g MTPExecutionGraph) CommitPlan(acceptance MTPAcceptance) (MTPKVCommitPlan, error) {
	if err := acceptance.Validate(); err != nil {
		return MTPKVCommitPlan{}, err
	}
	if acceptance.DraftedCount != len(g.DraftedTokens) {
		return MTPKVCommitPlan{}, fmt.Errorf("acceptance drafted count=%d, graph drafted count=%d", acceptance.DraftedCount, len(g.DraftedTokens))
	}
	keep := acceptance.KVKeepTokens()
	if keep <= 0 || keep > len(g.Verifier.Positions) {
		return MTPKVCommitPlan{}, fmt.Errorf("KV keep tokens=%d outside verifier positions len=%d", keep, len(g.Verifier.Positions))
	}
	return MTPKVCommitPlan{
		KeepTokens:   keep,
		Positions:    append([]int(nil), g.Verifier.Positions[:keep]...),
		OutputTokens: append([]int(nil), acceptance.OutputTokens...),
	}, nil
}
