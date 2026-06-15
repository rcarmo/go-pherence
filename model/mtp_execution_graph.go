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
	graph := MTPExecutionGraph{
		InputToken:      state.PreviousToken,
		DraftedTokens:   append([]int(nil), drafted...),
		StartPos:        startPos,
		DrafterSteps:    steps,
		Verifier:        verifier,
		MaxKVKeepTokens: len(drafted) + 1,
	}
	if err := graph.Validate(); err != nil {
		return MTPExecutionGraph{}, err
	}
	return graph, nil
}

func (g MTPExecutionGraph) Validate() error {
	if g.InputToken < 0 {
		return fmt.Errorf("MTP graph input token=%d out of range", g.InputToken)
	}
	if g.StartPos < 0 {
		return fmt.Errorf("MTP graph start position=%d out of range", g.StartPos)
	}
	if len(g.DraftedTokens) > maxMTPDraftCount {
		return fmt.Errorf("MTP graph draft count=%d out of range [0,%d]", len(g.DraftedTokens), maxMTPDraftCount)
	}
	for i, tok := range g.DraftedTokens {
		if tok < 0 {
			return fmt.Errorf("MTP graph drafted token %d=%d out of range", i, tok)
		}
	}
	if g.MaxKVKeepTokens != len(g.DraftedTokens)+1 {
		return fmt.Errorf("MTP graph max KV keep=%d, want drafted+1=%d", g.MaxKVKeepTokens, len(g.DraftedTokens)+1)
	}
	if len(g.DrafterSteps) != len(g.DraftedTokens) {
		return fmt.Errorf("MTP graph drafter steps=%d, drafted tokens=%d", len(g.DrafterSteps), len(g.DraftedTokens))
	}
	var activationWidth int
	var externalKVSeqLen int
	var externalKVLayers []int
	for i, step := range g.DrafterSteps {
		wantInput := g.InputToken
		if i > 0 {
			wantInput = g.DraftedTokens[i-1]
		}
		if step.Index != i || step.InputToken != wantInput || step.ActivationWidth <= 0 || step.ExternalKVSeqLen < 0 {
			return fmt.Errorf("MTP graph malformed drafter step %d: %+v want input=%d", i, step, wantInput)
		}
		if i == 0 {
			activationWidth = step.ActivationWidth
			externalKVSeqLen = step.ExternalKVSeqLen
			externalKVLayers = step.ExternalKVLayers
		} else {
			if step.ActivationWidth != activationWidth {
				return fmt.Errorf("MTP graph drafter step %d activation width=%d, want %d", i, step.ActivationWidth, activationWidth)
			}
			if step.ExternalKVSeqLen != externalKVSeqLen || !mtpSameInts(step.ExternalKVLayers, externalKVLayers) {
				return fmt.Errorf("MTP graph drafter step %d external KV view seq/layers=%d/%v, want %d/%v", i, step.ExternalKVSeqLen, step.ExternalKVLayers, externalKVSeqLen, externalKVLayers)
			}
		}
		seenExternalLayers := map[int]bool{}
		for j, layer := range step.ExternalKVLayers {
			if layer < 0 {
				return fmt.Errorf("MTP graph drafter step %d external KV layer %d=%d out of range", i, j, layer)
			}
			if seenExternalLayers[layer] {
				return fmt.Errorf("MTP graph drafter step %d reuses external KV layer %d", i, layer)
			}
			seenExternalLayers[layer] = true
		}
	}
	if g.Verifier.InputToken != g.InputToken || !mtpSameInts(g.Verifier.DraftedTokens, g.DraftedTokens) {
		return fmt.Errorf("MTP graph verifier/input mismatch graph input=%d drafted=%v verifier=%+v", g.InputToken, g.DraftedTokens, g.Verifier)
	}
	wantVerifierTokens, err := MTPVerifierTokens(g.InputToken, g.DraftedTokens)
	if err != nil {
		return err
	}
	if !mtpSameInts(g.Verifier.VerifierTokens, wantVerifierTokens) {
		return fmt.Errorf("MTP graph verifier tokens=%v, want %v", g.Verifier.VerifierTokens, wantVerifierTokens)
	}
	if g.Verifier.StartPos != g.StartPos {
		return fmt.Errorf("MTP graph verifier start=%d, graph start=%d", g.Verifier.StartPos, g.StartPos)
	}
	if len(g.Verifier.Positions) != len(g.DraftedTokens)+1 {
		return fmt.Errorf("MTP graph verifier positions=%d, want drafted+1=%d", len(g.Verifier.Positions), len(g.DraftedTokens)+1)
	}
	for i, pos := range g.Verifier.Positions {
		if pos != g.StartPos+i {
			return fmt.Errorf("MTP graph verifier position %d=%d, want %d", i, pos, g.StartPos+i)
		}
	}
	return nil
}

// CommitPlan returns the exact verifier KV window to retain after acceptance:
// accepted draft prefix plus the verifier bonus token. The returned positions
// are a prefix of the verifier pass positions and are safe to pass to KV commit
// helpers that retain appended positions.
func (g MTPExecutionGraph) CommitPlan(acceptance MTPAcceptance) (MTPKVCommitPlan, error) {
	if err := g.Validate(); err != nil {
		return MTPKVCommitPlan{}, err
	}
	if err := acceptance.Validate(); err != nil {
		return MTPKVCommitPlan{}, err
	}
	if acceptance.DraftedCount != len(g.DraftedTokens) {
		return MTPKVCommitPlan{}, fmt.Errorf("acceptance drafted count=%d, graph drafted count=%d", acceptance.DraftedCount, len(g.DraftedTokens))
	}
	for i, tok := range acceptance.AcceptedTokens {
		if i >= len(g.DraftedTokens) || tok != g.DraftedTokens[i] {
			return MTPKVCommitPlan{}, fmt.Errorf("acceptance token %d=%d does not match graph draft %d", i, tok, g.DraftedTokens[i])
		}
	}
	for i := 0; i < acceptance.AcceptedPrefixLen; i++ {
		if acceptance.OutputTokens[i] != g.DraftedTokens[i] {
			return MTPKVCommitPlan{}, fmt.Errorf("acceptance output token %d=%d does not match graph draft %d", i, acceptance.OutputTokens[i], g.DraftedTokens[i])
		}
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
