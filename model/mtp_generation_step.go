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
	verifierK, verifierV := s.KVCacheK, s.KVCacheV
	if s.CompressedKV != nil {
		var err error
		verifierK, verifierV, err = s.materializeCompressedKVForVerifier(len(s.Output))
		if err != nil {
			_ = s.Restore(cp)
			return MTPGraphDecodeStepResult{}, err
		}
	}
	step, err := s.Model.RunMTPMultiDraftSpeculativeStep(d, state, externalKV, len(s.Output), draftCount, verifierK, verifierV, stats)
	if err != nil {
		_ = s.Restore(cp)
		return MTPGraphDecodeStepResult{}, err
	}
	if s.CompressedKV != nil {
		if err := s.stageCompressedVerifierKV(verifierK, verifierV, len(s.Output)); err != nil {
			_ = s.Restore(cp)
			return MTPGraphDecodeStepResult{}, err
		}
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

func (s *CPUDecodeState) materializeCompressedKVForVerifier(startPos int) ([][]float32, [][]float32, error) {
	if s == nil || s.Model == nil {
		return nil, nil, fmt.Errorf("nil decode state/model")
	}
	if startPos < 0 {
		return nil, nil, fmt.Errorf("negative compressed verifier start position %d", startPos)
	}
	if len(s.KVDims) != len(s.Model.Layers) {
		return nil, nil, fmt.Errorf("compressed verifier KV dims=%d, want layers=%d", len(s.KVDims), len(s.Model.Layers))
	}
	if len(s.CompressedKV) != len(s.Model.Layers) {
		return nil, nil, fmt.Errorf("compressed verifier KV layers=%d, want %d", len(s.CompressedKV), len(s.Model.Layers))
	}
	k := make([][]float32, len(s.Model.Layers))
	v := make([][]float32, len(s.Model.Layers))
	for l, dim := range s.KVDims {
		if dim == 0 {
			continue
		}
		cache := s.CompressedKV[l]
		if cache == nil {
			return nil, nil, fmt.Errorf("compressed verifier layer %d cache is nil", l)
		}
		if cache.SeqLen() != startPos {
			return nil, nil, fmt.Errorf("compressed verifier layer %d seq len=%d, want start position %d", l, cache.SeqLen(), startPos)
		}
		want, ok := checkedProduct(startPos, dim)
		if !ok {
			return nil, nil, fmt.Errorf("compressed verifier layer %d length overflows start=%d dim=%d", l, startPos, dim)
		}
		kd := cache.GetK()
		vd := cache.GetV()
		if len(kd) != want || len(vd) != want {
			return nil, nil, fmt.Errorf("compressed verifier layer %d materialized K/V=%d/%d, want %d", l, len(kd), len(vd), want)
		}
		k[l] = append([]float32(nil), kd...)
		v[l] = append([]float32(nil), vd...)
	}
	return k, v, nil
}

func (s *CPUDecodeState) stageCompressedVerifierKV(kvCacheK, kvCacheV [][]float32, startPos int) error {
	if s == nil || s.Model == nil {
		return fmt.Errorf("nil decode state/model")
	}
	if startPos < 0 {
		return fmt.Errorf("negative compressed verifier start position %d", startPos)
	}
	if len(kvCacheK) != len(s.Model.Layers) || len(kvCacheV) != len(s.Model.Layers) || len(s.KVDims) != len(s.Model.Layers) || len(s.CompressedKV) != len(s.Model.Layers) {
		return fmt.Errorf("compressed verifier staging layers K/V/dims/caches=%d/%d/%d/%d, want %d", len(kvCacheK), len(kvCacheV), len(s.KVDims), len(s.CompressedKV), len(s.Model.Layers))
	}
	for l, dim := range s.KVDims {
		if dim == 0 {
			if len(kvCacheK[l]) != 0 || len(kvCacheV[l]) != 0 {
				return fmt.Errorf("compressed verifier shared layer %d staged unexpected K/V=%d/%d", l, len(kvCacheK[l]), len(kvCacheV[l]))
			}
			continue
		}
		cache := s.CompressedKV[l]
		if cache == nil {
			return fmt.Errorf("compressed verifier layer %d cache is nil", l)
		}
		base, ok := checkedProduct(startPos, dim)
		if !ok {
			return fmt.Errorf("compressed verifier layer %d base length overflows start=%d dim=%d", l, startPos, dim)
		}
		if len(kvCacheK[l]) < base || len(kvCacheV[l]) < base || (len(kvCacheK[l])-base)%dim != 0 || (len(kvCacheV[l])-base)%dim != 0 || len(kvCacheK[l]) != len(kvCacheV[l]) {
			return fmt.Errorf("compressed verifier layer %d staged K/V=%d/%d incompatible with base=%d dim=%d", l, len(kvCacheK[l]), len(kvCacheV[l]), base, dim)
		}
		if cache.SeqLen() != startPos {
			return fmt.Errorf("compressed verifier layer %d seq len=%d, want start position %d", l, cache.SeqLen(), startPos)
		}
		for off := base; off < len(kvCacheK[l]); off += dim {
			cache.Append(kvCacheK[l][off:off+dim], kvCacheV[l][off:off+dim])
		}
	}
	return nil
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
