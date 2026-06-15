package model

import "fmt"

type MTPGraphGenerationOptions struct {
	MaxTokens int
	Policy    MTPAdaptiveDraftPolicy
	Stats     MTPSpeculationStats
}

type MTPGraphGenerationResult struct {
	Output                     []int
	Stats                      MTPSpeculationStats
	FinalState                 MTPDrafterState
	Steps                      []MTPKVCommitPlan
	StepSummaries              []MTPGraphGenerationStepSummary
	GraphOutputTokens          int
	GreedyTailTokens           int
	Capabilities               MTPGraphCapabilities
	MissingForPublicGeneration []string
}

type MTPGraphGenerationStepSummary struct {
	InputToken        int
	DraftedTokens     []int
	VerifierTokens    []int
	Positions         []int
	AcceptedPrefixLen int
	BonusToken        int
	OutputTokens      []int
	AllDraftsAccepted bool
}

func (r MTPGraphGenerationResult) Validate(promptLen int) error {
	if promptLen < 0 || promptLen > len(r.Output) {
		return fmt.Errorf("MTP graph generation prompt len=%d outside output len=%d", promptLen, len(r.Output))
	}
	if r.GraphOutputTokens < 0 || r.GreedyTailTokens < 0 {
		return fmt.Errorf("MTP graph generation negative output accounting graph=%d greedy=%d", r.GraphOutputTokens, r.GreedyTailTokens)
	}
	if len(r.StepSummaries) != 0 && len(r.StepSummaries) != len(r.Steps) {
		return fmt.Errorf("MTP graph step summaries=%d, commit steps=%d", len(r.StepSummaries), len(r.Steps))
	}
	var graphCount int
	var graphTokens []int
	var summaryDrafted int
	var summaryVerified int
	for i, step := range r.Steps {
		if step.KeepTokens <= 0 || len(step.OutputTokens) != step.KeepTokens || len(step.Positions) != step.KeepTokens {
			return fmt.Errorf("MTP graph generation malformed commit step %d: %+v", i, step)
		}
		graphCount += len(step.OutputTokens)
		graphTokens = append(graphTokens, step.OutputTokens...)
		if len(r.StepSummaries) > 0 {
			summary := r.StepSummaries[i]
			if !mtpSameInts(summary.Positions, step.Positions) || !mtpSameInts(summary.OutputTokens, step.OutputTokens) {
				return fmt.Errorf("MTP graph summary %d does not match commit step summary=%+v commit=%+v", i, summary, step)
			}
			if len(summary.VerifierTokens) != len(summary.DraftedTokens)+1 || summary.VerifierTokens[0] != summary.InputToken {
				return fmt.Errorf("MTP graph summary %d verifier batch is not [input]+drafted: %+v", i, summary)
			}
			for j, tok := range summary.DraftedTokens {
				if summary.VerifierTokens[j+1] != tok {
					return fmt.Errorf("MTP graph summary %d verifier token %d=%d, want drafted %d", i, j+1, summary.VerifierTokens[j+1], tok)
				}
			}
			if len(summary.OutputTokens) != summary.AcceptedPrefixLen+1 || summary.BonusToken < 0 || summary.OutputTokens[summary.AcceptedPrefixLen] != summary.BonusToken {
				return fmt.Errorf("MTP graph summary %d invalid acceptance/output accounting: %+v", i, summary)
			}
			if summary.AcceptedPrefixLen < 0 || summary.AcceptedPrefixLen > len(summary.DraftedTokens) {
				return fmt.Errorf("MTP graph summary %d accepted prefix=%d drafted=%d", i, summary.AcceptedPrefixLen, len(summary.DraftedTokens))
			}
			if summary.AllDraftsAccepted != (summary.AcceptedPrefixLen == len(summary.DraftedTokens)) {
				return fmt.Errorf("MTP graph summary %d all-accepted=%v inconsistent with accepted=%d drafted=%d", i, summary.AllDraftsAccepted, summary.AcceptedPrefixLen, len(summary.DraftedTokens))
			}
			summaryDrafted += len(summary.DraftedTokens)
			summaryVerified += summary.AcceptedPrefixLen
		}
	}
	if len(r.StepSummaries) > 0 {
		if r.Stats.Steps != len(r.StepSummaries) {
			return fmt.Errorf("MTP stats steps=%d, summary steps=%d", r.Stats.Steps, len(r.StepSummaries))
		}
		if r.Stats.DraftedTokens != summaryDrafted {
			return fmt.Errorf("MTP stats drafted tokens=%d, summary drafted=%d", r.Stats.DraftedTokens, summaryDrafted)
		}
		if r.Stats.VerifiedTokens != summaryVerified {
			return fmt.Errorf("MTP stats verified tokens=%d, summary verified=%d", r.Stats.VerifiedTokens, summaryVerified)
		}
		if r.Stats.BonusTokens != len(r.StepSummaries) {
			return fmt.Errorf("MTP stats bonus tokens=%d, summary steps=%d", r.Stats.BonusTokens, len(r.StepSummaries))
		}
	}
	if graphCount != r.GraphOutputTokens {
		return fmt.Errorf("MTP graph output accounting=%d, commit outputs=%d", r.GraphOutputTokens, graphCount)
	}
	if len(graphTokens) > 0 {
		end := promptLen + len(graphTokens)
		if end > len(r.Output) || !mtpSameInts(r.Output[promptLen:end], graphTokens) {
			return fmt.Errorf("MTP graph output tokens do not match output stream: output=%v graph=%v promptLen=%d", r.Output, graphTokens, promptLen)
		}
	}
	generated := len(r.Output) - promptLen
	if generated != r.GraphOutputTokens+r.GreedyTailTokens {
		return fmt.Errorf("MTP generated tokens=%d, graph+greedy=%d+%d", generated, r.GraphOutputTokens, r.GreedyTailTokens)
	}
	if r.Stats.OutputTokens != r.GraphOutputTokens {
		return fmt.Errorf("MTP stats output tokens=%d, graph output tokens=%d", r.Stats.OutputTokens, r.GraphOutputTokens)
	}
	return nil
}

// NewCPUDecodeStateFromMTPPromptContext creates a graph-decode state from a
// prompt context produced by BuildMTPPromptContext. It copies tokens and KV so a
// failed speculative step can restore safely without mutating the prompt context.
func NewCPUDecodeStateFromMTPPromptContext(m *LlamaModel, ctx MTPPromptContext, maxTokens int) (*CPUDecodeState, error) {
	if m == nil {
		return nil, fmt.Errorf("nil model")
	}
	if maxTokens < 0 {
		return nil, fmt.Errorf("maxTokens=%d must be >= 0", maxTokens)
	}
	if len(ctx.Tokens) == 0 || ctx.SeqLen != len(ctx.Tokens) {
		return nil, fmt.Errorf("invalid MTP prompt context tokens=%d seqLen=%d", len(ctx.Tokens), ctx.SeqLen)
	}
	if ctx.PreviousToken != ctx.Tokens[len(ctx.Tokens)-1] {
		return nil, fmt.Errorf("prompt context previous token=%d, want final token %d", ctx.PreviousToken, ctx.Tokens[len(ctx.Tokens)-1])
	}
	if m.Config.HiddenSize <= 0 || len(ctx.Activation) != m.Config.HiddenSize {
		return nil, fmt.Errorf("prompt context activation len=%d, want hidden=%d", len(ctx.Activation), m.Config.HiddenSize)
	}
	state, err := NewCPUDecodeStateForSpeculative(m, ctx.Tokens, maxTokens, "mtp-graph")
	if err != nil {
		return nil, err
	}
	if len(ctx.KVCacheK) != len(state.KVCacheK) || len(ctx.KVCacheV) != len(state.KVCacheV) {
		return nil, fmt.Errorf("prompt context KV layers K/V=%d/%d, want %d", len(ctx.KVCacheK), len(ctx.KVCacheV), len(state.KVCacheK))
	}
	for l, dim := range state.KVDims {
		if dim <= 0 {
			if len(ctx.KVCacheK[l]) != 0 || len(ctx.KVCacheV[l]) != 0 {
				return nil, fmt.Errorf("prompt context shared/non-KV layer %d has K/V entries %d/%d", l, len(ctx.KVCacheK[l]), len(ctx.KVCacheV[l]))
			}
			continue
		}
		want := ctx.SeqLen * dim
		if len(ctx.KVCacheK[l]) != want || len(ctx.KVCacheV[l]) != want {
			return nil, fmt.Errorf("prompt context layer %d KV K/V=%d/%d, want %d", l, len(ctx.KVCacheK[l]), len(ctx.KVCacheV[l]), want)
		}
		state.KVCacheK[l] = append(state.KVCacheK[l], ctx.KVCacheK[l]...)
		state.KVCacheV[l] = append(state.KVCacheV[l], ctx.KVCacheV[l]...)
	}
	return state, nil
}

// GenerateMTPGraphFromPromptContext runs the internal graph-backed MTP cycle
// from a prefilled prompt context. It is the model-level production contract for
// future CLI/API wiring; current real Gemma4 verifier limitations still apply.
func (m *LlamaModel) GenerateMTPGraphFromPromptContext(d *Gemma4MTPDrafter, ctx MTPPromptContext, externalKV *MTPDrafterExternalKV, opts MTPGraphGenerationOptions) (MTPGraphGenerationResult, error) {
	if m == nil {
		return MTPGraphGenerationResult{}, fmt.Errorf("nil model")
	}
	if d == nil {
		return MTPGraphGenerationResult{}, fmt.Errorf("nil drafter")
	}
	if opts.MaxTokens < 0 {
		return MTPGraphGenerationResult{}, fmt.Errorf("maxTokens=%d must be >= 0", opts.MaxTokens)
	}
	decode, err := NewCPUDecodeStateFromMTPPromptContext(m, ctx, opts.MaxTokens)
	if err != nil {
		return MTPGraphGenerationResult{}, err
	}
	state, err := NewMTPDrafterState(ctx.PreviousToken, ctx.Activation, d.BackboneHiddenSize)
	if err != nil {
		return MTPGraphGenerationResult{}, err
	}
	stats := opts.Stats
	commits := make([]MTPKVCommitPlan, 0)
	summaries := make([]MTPGraphGenerationStepSummary, 0)
	graphOutputTokens := 0
	greedyTailTokens := 0
	for len(decode.Output)-len(ctx.Tokens) < opts.MaxTokens {
		remaining := opts.MaxTokens - (len(decode.Output) - len(ctx.Tokens))
		if remaining <= 1 {
			// A graph verifier pass with G drafts can emit up to G+1 tokens, so a
			// one-token tail is ordinary greedy decode territory. This mirrors
			// speculative generation fallback behavior while preserving the graph
			// contract for multi-token steps.
			if _, err := decode.DecodeOneGreedy(); err != nil {
				return MTPGraphGenerationResult{}, err
			}
			greedyTailTokens++
			break
		}
		stepExternalKV, err := mtpExternalKVForDecodeState(decode, externalKV)
		if err != nil {
			return MTPGraphGenerationResult{}, err
		}
		step, err := decode.RunMTPGraphDecodeStep(d, state, stepExternalKV, MTPGraphDecodeStepOptions{RemainingTokens: remaining, Policy: opts.Policy}, stats)
		if err != nil {
			return MTPGraphGenerationResult{}, err
		}
		state = step.FinalState
		stats = step.Stats
		commits = append(commits, step.Commit)
		summaries = append(summaries, newMTPGraphGenerationStepSummary(step))
		graphOutputTokens += len(step.Commit.OutputTokens)
		if len(step.Commit.OutputTokens) == 0 {
			break
		}
	}
	caps := Gemma4MTPGraphCapabilities()
	result := MTPGraphGenerationResult{Output: append([]int(nil), decode.Output...), Stats: stats, FinalState: state, Steps: commits, StepSummaries: summaries, GraphOutputTokens: graphOutputTokens, GreedyTailTokens: greedyTailTokens, Capabilities: caps, MissingForPublicGeneration: caps.MissingForPublicGeneration()}
	if err := result.Validate(len(ctx.Tokens)); err != nil {
		return MTPGraphGenerationResult{}, err
	}
	return result, nil
}

func newMTPGraphGenerationStepSummary(step MTPGraphDecodeStepResult) MTPGraphGenerationStepSummary {
	return MTPGraphGenerationStepSummary{
		InputToken:        step.Step.Graph.InputToken,
		DraftedTokens:     append([]int(nil), step.Step.Drafts.Tokens...),
		VerifierTokens:    append([]int(nil), step.Step.Plan.VerifierTokens...),
		Positions:         append([]int(nil), step.Commit.Positions...),
		AcceptedPrefixLen: step.Step.Verifier.Acceptance.AcceptedPrefixLen,
		BonusToken:        step.Step.Verifier.Acceptance.BonusToken,
		OutputTokens:      append([]int(nil), step.Commit.OutputTokens...),
		AllDraftsAccepted: step.Step.Verifier.Acceptance.AllDraftsAccepted,
	}
}

func mtpExternalKVForDecodeState(decode *CPUDecodeState, base *MTPDrafterExternalKV) (*MTPDrafterExternalKV, error) {
	if base == nil {
		return nil, nil
	}
	if decode == nil {
		return nil, fmt.Errorf("nil decode state")
	}
	if len(decode.KVCacheK) != len(base.K) || len(decode.KVCacheV) != len(base.V) {
		return nil, fmt.Errorf("decode KV layers K/V=%d/%d, base K/V=%d/%d", len(decode.KVCacheK), len(decode.KVCacheV), len(base.K), len(base.V))
	}
	return &MTPDrafterExternalKV{
		K:            decode.KVCacheK,
		V:            decode.KVCacheV,
		SourceLayers: append([]int(nil), base.SourceLayers...),
		SeqLen:       len(decode.Output),
	}, nil
}
