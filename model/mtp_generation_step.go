package model

import "fmt"

// MTPAdaptiveDraftPolicy chooses the next draft count for Gemma4 MTP generation.
// It is intentionally small and deterministic: callers can replace it later
// without changing the MTPExecutionGraph/commit contracts.
type MTPAdaptiveDraftPolicy struct {
	MinDrafts          int
	InitialDrafts      int
	MaxDrafts          int
	IncreaseAcceptance float64
	DecreaseAcceptance float64
}

func (p MTPAdaptiveDraftPolicy) Normalize() MTPAdaptiveDraftPolicy {
	if p.MinDrafts <= 0 {
		p.MinDrafts = 1
	}
	if p.InitialDrafts <= 0 {
		p.InitialDrafts = p.MinDrafts
	}
	if p.MaxDrafts <= 0 || p.MaxDrafts > maxMTPDraftCount {
		p.MaxDrafts = maxMTPDraftCount
	}
	if p.MinDrafts > p.MaxDrafts {
		p.MinDrafts = p.MaxDrafts
	}
	if p.InitialDrafts < p.MinDrafts {
		p.InitialDrafts = p.MinDrafts
	}
	if p.InitialDrafts > p.MaxDrafts {
		p.InitialDrafts = p.MaxDrafts
	}
	if p.IncreaseAcceptance <= 0 {
		p.IncreaseAcceptance = 0.75
	}
	if p.DecreaseAcceptance <= 0 {
		p.DecreaseAcceptance = 0.40
	}
	if p.DecreaseAcceptance > p.IncreaseAcceptance {
		p.DecreaseAcceptance = p.IncreaseAcceptance
	}
	return p
}

// NextDraftCount returns how many draft tokens to propose without exceeding the
// remaining output budget. Because a verifier pass can emit accepted drafts plus
// one bonus token, a speculative pass with G drafts can emit up to G+1 tokens;
// therefore remainingOutputTokens <= 1 returns 0 and should fall back to normal
// decode.
func (p MTPAdaptiveDraftPolicy) NextDraftCount(remainingOutputTokens int, stats MTPSpeculationStats) int {
	if remainingOutputTokens <= 1 {
		return 0
	}
	p = p.Normalize()
	limit := remainingOutputTokens - 1
	if limit > p.MaxDrafts {
		limit = p.MaxDrafts
	}
	if limit <= 0 {
		return 0
	}
	draft := p.InitialDrafts
	if stats.Steps > 0 && stats.DraftedTokens > 0 {
		draft = (stats.DraftedTokens + stats.Steps - 1) / stats.Steps // ceil average proposal length
		rate := stats.AcceptanceRate()
		if rate >= p.IncreaseAcceptance {
			draft++
		} else if rate <= p.DecreaseAcceptance {
			draft--
		}
	}
	if draft < p.MinDrafts {
		draft = p.MinDrafts
	}
	if draft > limit {
		draft = limit
	}
	return draft
}

type MTPGraphDecodeStepOptions struct {
	RemainingTokens int
	DraftCount      int
	Policy          MTPAdaptiveDraftPolicy
}

type MTPGraphDecodeStepResult struct {
	Step       MTPMultiDraftSpeculativeResult
	Commit     MTPKVCommitPlan
	Stats      MTPSpeculationStats
	FinalState MTPDrafterState
}

// RunMTPGraphDecodeStep is the internal production-cycle contract for Gemma4
// MTP generation: choose/validate G, draft G tokens, verify them in one graph,
// graph-commit accepted-prefix+bonus KV, and append exactly the committed output
// tokens to the decode state.
func (s *CPUDecodeState) RunMTPGraphDecodeStep(d *Gemma4MTPDrafter, state MTPDrafterState, externalKV *MTPDrafterExternalKV, opts MTPGraphDecodeStepOptions, stats MTPSpeculationStats) (MTPGraphDecodeStepResult, error) {
	if s == nil || s.Model == nil {
		return MTPGraphDecodeStepResult{}, fmt.Errorf("nil decode state/model")
	}
	if opts.RemainingTokens <= 0 {
		return MTPGraphDecodeStepResult{}, fmt.Errorf("remaining tokens %d out of range", opts.RemainingTokens)
	}
	draftCount := opts.DraftCount
	if draftCount == 0 {
		draftCount = opts.Policy.NextDraftCount(opts.RemainingTokens, stats)
	}
	if draftCount <= 0 {
		return MTPGraphDecodeStepResult{}, fmt.Errorf("remaining tokens %d insufficient for MTP draft+bonus", opts.RemainingTokens)
	}
	if draftCount >= opts.RemainingTokens {
		return MTPGraphDecodeStepResult{}, fmt.Errorf("draft count %d would exceed remaining output budget %d including bonus", draftCount, opts.RemainingTokens)
	}
	cp := s.Checkpoint()
	step, err := s.Model.RunMTPMultiDraftSpeculativeStep(d, state, externalKV, len(s.Output), draftCount, s.KVCacheK, s.KVCacheV, stats)
	if err != nil {
		_ = s.Restore(cp)
		return MTPGraphDecodeStepResult{}, err
	}
	commit, err := s.CommitGraphAccepted(cp, step.Graph, step.Verifier)
	if err != nil {
		_ = s.Restore(cp)
		return MTPGraphDecodeStepResult{}, err
	}
	nextState, err := newMTPDrafterStateFromVerifierCommit(d, step.Verifier, commit)
	if err != nil {
		_ = s.Restore(cp)
		return MTPGraphDecodeStepResult{}, err
	}
	return MTPGraphDecodeStepResult{Step: step, Commit: commit, Stats: step.Stats, FinalState: nextState}, nil
}

func newMTPDrafterStateFromVerifierCommit(d *Gemma4MTPDrafter, verifier MTPVerifierResult, commit MTPKVCommitPlan) (MTPDrafterState, error) {
	if d == nil {
		return MTPDrafterState{}, fmt.Errorf("nil drafter")
	}
	if len(commit.OutputTokens) == 0 {
		return MTPDrafterState{}, fmt.Errorf("empty MTP commit output tokens")
	}
	activation, err := verifier.CommittedActivation()
	if err != nil {
		return MTPDrafterState{}, err
	}
	return NewMTPDrafterState(commit.OutputTokens[len(commit.OutputTokens)-1], activation, d.BackboneHiddenSize)
}
