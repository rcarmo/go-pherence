package model

import "fmt"

type MTPGraphGenerationOptions struct {
	MaxTokens int
	Policy    MTPAdaptiveDraftPolicy
	Stats     MTPSpeculationStats
}

type MTPGraphGenerationResult struct {
	Output                     []int
	VocabSize                  int
	HiddenSize                 int
	RequestedMaxTokens         int
	InitialStats               MTPSpeculationStats
	Stats                      MTPSpeculationStats
	FinalState                 MTPDrafterState
	FinalStateOutputLen        int // number of Output tokens covered by FinalState; greedy-tail fallback is not covered
	Steps                      []MTPKVCommitPlan
	StepSummaries              []MTPGraphGenerationStepSummary
	GraphOutputTokens          int
	GreedyTailTokens           int
	Capabilities               MTPGraphCapabilities
	MissingForPublicGeneration []string
}

type MTPGraphGenerationStepSummary struct {
	InputToken           int
	DraftedTokens        []int
	VerifierTokens       []int // verifier input batch: [input] + drafted
	VerifierOutputTokens []int // greedy verifier outputs from logits, len drafted+1
	VerifierPositions    []int // full verifier batch positions, len drafted+1
	Positions            []int // committed positions, accepted prefix + bonus
	AcceptedPrefixLen    int
	BonusToken           int
	OutputTokens         []int
	AllDraftsAccepted    bool
}

func (r MTPGraphGenerationResult) Validate(promptLen int) error {
	if promptLen < 0 || promptLen > len(r.Output) {
		return fmt.Errorf("MTP graph generation prompt len=%d outside output len=%d", promptLen, len(r.Output))
	}
	if r.GraphOutputTokens < 0 || r.GreedyTailTokens < 0 {
		return fmt.Errorf("MTP graph generation negative output accounting graph=%d greedy=%d", r.GraphOutputTokens, r.GreedyTailTokens)
	}
	if r.RequestedMaxTokens < 0 {
		return fmt.Errorf("MTP graph generation requested max tokens=%d out of range", r.RequestedMaxTokens)
	}
	if r.VocabSize < 0 {
		return fmt.Errorf("MTP graph generation vocab size=%d out of range", r.VocabSize)
	}
	if len(r.Output) > 0 && r.VocabSize == 0 {
		return fmt.Errorf("MTP graph generation vocab size is unset for non-empty output len=%d", len(r.Output))
	}
	if r.HiddenSize < 0 {
		return fmt.Errorf("MTP graph generation hidden size=%d out of range", r.HiddenSize)
	}
	if len(r.Output) > 0 && r.HiddenSize == 0 {
		return fmt.Errorf("MTP graph generation hidden size is unset for non-empty output len=%d", len(r.Output))
	}
	if err := validateMTPGraphSummaryTokens(-1, "output", r.Output, r.VocabSize); err != nil {
		return err
	}
	if len(r.StepSummaries) != len(r.Steps) {
		return fmt.Errorf("MTP graph step summaries=%d, commit steps=%d", len(r.StepSummaries), len(r.Steps))
	}
	if err := r.Capabilities.Validate(); err != nil {
		return err
	}
	if !mtpSameStringSet(r.MissingForPublicGeneration, r.Capabilities.MissingForPublicGeneration()) {
		return fmt.Errorf("MTP public-generation blockers=%v, want capabilities blockers=%v", r.MissingForPublicGeneration, r.Capabilities.MissingForPublicGeneration())
	}
	var graphCount int
	var graphTokens []int
	var summaryDrafted int
	var summaryVerified int
	statsDelta, err := mtpStatsDelta(r.Stats, r.InitialStats)
	if err != nil {
		return err
	}
	streamCursor := promptLen
	for i, step := range r.Steps {
		if step.KeepTokens <= 0 || len(step.OutputTokens) != step.KeepTokens || len(step.Positions) != step.KeepTokens {
			return fmt.Errorf("MTP graph generation malformed commit step %d: %+v", i, step)
		}
		graphCount += len(step.OutputTokens)
		graphTokens = append(graphTokens, step.OutputTokens...)
		summary := r.StepSummaries[i]
		if streamCursor <= 0 || streamCursor > len(r.Output) {
			return fmt.Errorf("MTP graph summary %d stream cursor=%d outside output len=%d", i, streamCursor, len(r.Output))
		}
		if summary.InputToken < 0 || (r.VocabSize > 0 && summary.InputToken >= r.VocabSize) {
			return fmt.Errorf("MTP graph summary %d input token=%d out of range", i, summary.InputToken)
		}
		if summary.InputToken != r.Output[streamCursor-1] {
			return fmt.Errorf("MTP graph summary %d input token=%d, want output cursor token %d at output[%d]", i, summary.InputToken, r.Output[streamCursor-1], streamCursor-1)
		}
		if err := validateMTPGraphSummaryTokens(i, "drafted", summary.DraftedTokens, r.VocabSize); err != nil {
			return err
		}
		if err := validateMTPGraphSummaryTokens(i, "verifier", summary.VerifierTokens, r.VocabSize); err != nil {
			return err
		}
		if err := validateMTPGraphSummaryTokens(i, "verifier output", summary.VerifierOutputTokens, r.VocabSize); err != nil {
			return err
		}
		if err := validateMTPGraphSummaryTokens(i, "output", summary.OutputTokens, r.VocabSize); err != nil {
			return err
		}
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
		for j := 0; j < summary.AcceptedPrefixLen; j++ {
			if summary.OutputTokens[j] != summary.DraftedTokens[j] {
				return fmt.Errorf("MTP graph summary %d output token %d=%d, want accepted draft %d", i, j, summary.OutputTokens[j], summary.DraftedTokens[j])
			}
		}
		if summary.AllDraftsAccepted != (summary.AcceptedPrefixLen == len(summary.DraftedTokens)) {
			return fmt.Errorf("MTP graph summary %d all-accepted=%v inconsistent with accepted=%d drafted=%d", i, summary.AllDraftsAccepted, summary.AcceptedPrefixLen, len(summary.DraftedTokens))
		}
		if len(summary.VerifierPositions) != len(summary.DraftedTokens)+1 {
			return fmt.Errorf("MTP graph summary %d verifier positions len=%d, want drafted+1=%d", i, len(summary.VerifierPositions), len(summary.DraftedTokens)+1)
		}
		for j, pos := range summary.VerifierPositions {
			if pos < 0 {
				return fmt.Errorf("MTP graph summary %d verifier position %d=%d out of range", i, j, pos)
			}
			if j == 0 && pos != streamCursor {
				return fmt.Errorf("MTP graph summary %d first verifier position=%d, want stream cursor=%d", i, pos, streamCursor)
			}
			if j > 0 && pos != summary.VerifierPositions[j-1]+1 {
				return fmt.Errorf("MTP graph summary %d verifier positions not contiguous: %v", i, summary.VerifierPositions)
			}
		}
		if !mtpSameInts(summary.Positions, summary.VerifierPositions[:len(summary.Positions)]) {
			return fmt.Errorf("MTP graph summary %d committed positions=%v are not verifier prefix %v", i, summary.Positions, summary.VerifierPositions)
		}
		if len(summary.VerifierOutputTokens) != len(summary.DraftedTokens)+1 {
			return fmt.Errorf("MTP graph summary %d verifier outputs len=%d, want drafted+1=%d", i, len(summary.VerifierOutputTokens), len(summary.DraftedTokens)+1)
		}
		maxAccepted := 0
		for maxAccepted < len(summary.DraftedTokens) && summary.VerifierOutputTokens[maxAccepted] == summary.DraftedTokens[maxAccepted] {
			maxAccepted++
		}
		if summary.AcceptedPrefixLen != maxAccepted {
			return fmt.Errorf("MTP graph summary %d accepted prefix=%d, want maximal verifier/draft match=%d", i, summary.AcceptedPrefixLen, maxAccepted)
		}
		for j := 0; j < summary.AcceptedPrefixLen; j++ {
			if summary.VerifierOutputTokens[j] != summary.DraftedTokens[j] {
				return fmt.Errorf("MTP graph summary %d verifier output %d=%d, want accepted draft %d", i, j, summary.VerifierOutputTokens[j], summary.DraftedTokens[j])
			}
		}
		if summary.BonusToken != summary.VerifierOutputTokens[summary.AcceptedPrefixLen] {
			return fmt.Errorf("MTP graph summary %d bonus token=%d, want verifier output %d", i, summary.BonusToken, summary.VerifierOutputTokens[summary.AcceptedPrefixLen])
		}
		streamCursor += len(summary.OutputTokens)
		summaryDrafted += len(summary.DraftedTokens)
		summaryVerified += summary.AcceptedPrefixLen
	}
	if statsDelta.Steps != len(r.StepSummaries) {
		return fmt.Errorf("MTP stats steps delta=%d, summary steps=%d", statsDelta.Steps, len(r.StepSummaries))
	}
	if statsDelta.DraftedTokens != summaryDrafted {
		return fmt.Errorf("MTP stats drafted tokens delta=%d, summary drafted=%d", statsDelta.DraftedTokens, summaryDrafted)
	}
	if statsDelta.VerifiedTokens != summaryVerified {
		return fmt.Errorf("MTP stats verified tokens delta=%d, summary verified=%d", statsDelta.VerifiedTokens, summaryVerified)
	}
	if statsDelta.BonusTokens != len(r.StepSummaries) {
		return fmt.Errorf("MTP stats bonus tokens delta=%d, summary steps=%d", statsDelta.BonusTokens, len(r.StepSummaries))
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
	if generated != r.RequestedMaxTokens {
		return fmt.Errorf("MTP generated tokens=%d, requested max tokens=%d", generated, r.RequestedMaxTokens)
	}
	if generated != r.GraphOutputTokens+r.GreedyTailTokens {
		return fmt.Errorf("MTP generated tokens=%d, graph+greedy=%d+%d", generated, r.GraphOutputTokens, r.GreedyTailTokens)
	}
	if statsDelta.OutputTokens != r.GraphOutputTokens {
		return fmt.Errorf("MTP stats output tokens delta=%d, graph output tokens=%d", statsDelta.OutputTokens, r.GraphOutputTokens)
	}
	if len(r.Output) > 0 && r.FinalStateOutputLen == 0 {
		return fmt.Errorf("MTP final state output len is unset for non-empty output len=%d", len(r.Output))
	}
	if r.FinalStateOutputLen != 0 {
		wantFinalStateOutputLen := promptLen + r.GraphOutputTokens
		if r.FinalStateOutputLen != wantFinalStateOutputLen {
			return fmt.Errorf("MTP final state output len=%d, want prompt+graph output=%d", r.FinalStateOutputLen, wantFinalStateOutputLen)
		}
		if r.FinalStateOutputLen < 0 || r.FinalStateOutputLen > len(r.Output) {
			return fmt.Errorf("MTP final state output len=%d outside output len=%d", r.FinalStateOutputLen, len(r.Output))
		}
		if r.FinalStateOutputLen > 0 {
			wantToken := r.Output[r.FinalStateOutputLen-1]
			if r.FinalState.PreviousToken != wantToken {
				return fmt.Errorf("MTP final state previous token=%d, want output[%d]=%d", r.FinalState.PreviousToken, r.FinalStateOutputLen-1, wantToken)
			}
			if r.HiddenSize > 0 && len(r.FinalState.Activation) != r.HiddenSize {
				return fmt.Errorf("MTP final state activation len=%d, want hidden size=%d", len(r.FinalState.Activation), r.HiddenSize)
			}
		}
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
	finalStateOutputLen := len(ctx.Tokens)
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
		finalStateOutputLen = len(decode.Output)
		stats = step.Stats
		commits = append(commits, step.Commit)
		summaries = append(summaries, newMTPGraphGenerationStepSummary(step))
		graphOutputTokens += len(step.Commit.OutputTokens)
		if len(step.Commit.OutputTokens) == 0 {
			break
		}
	}
	caps := Gemma4MTPGraphCapabilities()
	result := MTPGraphGenerationResult{Output: append([]int(nil), decode.Output...), VocabSize: m.Config.VocabSize, HiddenSize: m.Config.HiddenSize, RequestedMaxTokens: opts.MaxTokens, InitialStats: opts.Stats, Stats: stats, FinalState: state, FinalStateOutputLen: finalStateOutputLen, Steps: commits, StepSummaries: summaries, GraphOutputTokens: graphOutputTokens, GreedyTailTokens: greedyTailTokens, Capabilities: caps, MissingForPublicGeneration: caps.MissingForPublicGeneration()}
	if err := result.Validate(len(ctx.Tokens)); err != nil {
		return MTPGraphGenerationResult{}, err
	}
	return result, nil
}

func newMTPGraphGenerationStepSummary(step MTPGraphDecodeStepResult) MTPGraphGenerationStepSummary {
	verifierOutputs := make([]int, 0, len(step.Step.Verifier.Logits))
	for _, logits := range step.Step.Verifier.Logits {
		id, _, err := ArgmaxLogits(logits)
		if err != nil {
			id = -1
		}
		verifierOutputs = append(verifierOutputs, id)
	}
	return MTPGraphGenerationStepSummary{
		InputToken:           step.Step.Graph.InputToken,
		DraftedTokens:        append([]int(nil), step.Step.Drafts.Tokens...),
		VerifierTokens:       append([]int(nil), step.Step.Plan.VerifierTokens...),
		VerifierOutputTokens: verifierOutputs,
		VerifierPositions:    append([]int(nil), step.Step.Plan.Positions...),
		Positions:            append([]int(nil), step.Commit.Positions...),
		AcceptedPrefixLen:    step.Step.Verifier.Acceptance.AcceptedPrefixLen,
		BonusToken:           step.Step.Verifier.Acceptance.BonusToken,
		OutputTokens:         append([]int(nil), step.Commit.OutputTokens...),
		AllDraftsAccepted:    step.Step.Verifier.Acceptance.AllDraftsAccepted,
	}
}

func mtpStatsDelta(total, initial MTPSpeculationStats) (MTPSpeculationStats, error) {
	if total.Steps < 0 || total.DraftedTokens < 0 || total.VerifiedTokens < 0 || total.BonusTokens < 0 || total.OutputTokens < 0 ||
		initial.Steps < 0 || initial.DraftedTokens < 0 || initial.VerifiedTokens < 0 || initial.BonusTokens < 0 || initial.OutputTokens < 0 {
		return MTPSpeculationStats{}, fmt.Errorf("invalid MTP stats counters total=%+v initial=%+v", total, initial)
	}
	if total.Steps < initial.Steps || total.DraftedTokens < initial.DraftedTokens || total.VerifiedTokens < initial.VerifiedTokens || total.BonusTokens < initial.BonusTokens || total.OutputTokens < initial.OutputTokens {
		return MTPSpeculationStats{}, fmt.Errorf("MTP stats counters moved backwards total=%+v initial=%+v", total, initial)
	}
	return MTPSpeculationStats{
		Steps:          total.Steps - initial.Steps,
		DraftedTokens:  total.DraftedTokens - initial.DraftedTokens,
		VerifiedTokens: total.VerifiedTokens - initial.VerifiedTokens,
		BonusTokens:    total.BonusTokens - initial.BonusTokens,
		OutputTokens:   total.OutputTokens - initial.OutputTokens,
	}, nil
}

func mtpSameStringSet(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	seen := make(map[string]int, len(a))
	for _, x := range a {
		seen[x]++
	}
	for _, x := range b {
		if seen[x] == 0 {
			return false
		}
		seen[x]--
	}
	return true
}

func validateMTPGraphSummaryTokens(step int, label string, tokens []int, vocab int) error {
	for i, tok := range tokens {
		if tok < 0 || (vocab > 0 && tok >= vocab) {
			return fmt.Errorf("MTP graph summary %d %s token %d=%d out of range", step, label, i, tok)
		}
	}
	return nil
}

func mtpExternalKVForDecodeState(decode *CPUDecodeState, base *MTPDrafterExternalKV) (*MTPDrafterExternalKV, error) {
	if base == nil {
		return nil, nil
	}
	if decode == nil {
		return nil, fmt.Errorf("nil decode state")
	}
	kvK, kvV := decode.KVCacheK, decode.KVCacheV
	if decode.CompressedKV != nil {
		var err error
		kvK, kvV, err = decode.materializeCompressedKVForVerifier(len(decode.Output))
		if err != nil {
			return nil, err
		}
	}
	if len(kvK) != len(base.K) || len(kvV) != len(base.V) {
		return nil, fmt.Errorf("decode KV layers K/V=%d/%d, base K/V=%d/%d", len(kvK), len(kvV), len(base.K), len(base.V))
	}
	return &MTPDrafterExternalKV{
		K:            kvK,
		V:            kvV,
		SourceLayers: append([]int(nil), base.SourceLayers...),
		SeqLen:       len(decode.Output),
	}, nil
}
