package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestNewCPUDecodeStateFromMTPPromptContext(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 1, NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	ctx := MTPPromptContext{Tokens: []int{1, 2}, PreviousToken: 2, Activation: []float32{0.5, 0.25}, SeqLen: 2, KVCacheK: [][]float32{{1, 2, 3, 4}}, KVCacheV: [][]float32{{5, 6, 7, 8}}}
	st, err := NewCPUDecodeStateFromMTPPromptContext(m, ctx, 3)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(st.Output, ctx.Tokens) || !sameFloat32s(st.KVCacheK[0], ctx.KVCacheK[0]) || !sameFloat32s(st.KVCacheV[0], ctx.KVCacheV[0]) {
		t.Fatalf("state output=%v K=%v V=%v", st.Output, st.KVCacheK, st.KVCacheV)
	}
	ctx.KVCacheK[0][0] = 99
	if st.KVCacheK[0][0] == 99 {
		t.Fatal("decode state aliases prompt context KV")
	}
}

func TestNewCPUDecodeStateFromMTPPromptContextValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 1, NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	base := MTPPromptContext{Tokens: []int{1, 2}, PreviousToken: 2, Activation: []float32{0.5, 0.25}, SeqLen: 2, KVCacheK: [][]float32{{1, 2, 3, 4}}, KVCacheV: [][]float32{{5, 6, 7, 8}}}
	if _, err := NewCPUDecodeStateFromMTPPromptContext(nil, base, 1); err == nil {
		t.Fatal("accepted nil model")
	}
	bad := base
	bad.SeqLen = 1
	if _, err := NewCPUDecodeStateFromMTPPromptContext(m, bad, 1); err == nil {
		t.Fatal("accepted bad seqLen")
	}
	bad = base
	bad.Activation = []float32{1}
	if _, err := NewCPUDecodeStateFromMTPPromptContext(m, bad, 1); err == nil {
		t.Fatal("accepted bad activation")
	}
	bad = base
	bad.KVCacheK = [][]float32{{1, 2}}
	if _, err := NewCPUDecodeStateFromMTPPromptContext(m, bad, 1); err == nil {
		t.Fatal("accepted bad KV width")
	}
	maxInt := int(^uint(0) >> 1)
	overflowModel := &LlamaModel{Config: LlamaConfig{VocabSize: 4, HiddenSize: 2, NumLayers: 1, NumKVHeads: maxInt/2 + 1, HeadDim: 3}, Layers: []LlamaLayer{{HasKV: true}}}
	overflow := MTPPromptContext{Tokens: []int{1, 2}, PreviousToken: 2, Activation: []float32{0.5, 0.25}, SeqLen: 2, KVCacheK: [][]float32{{}}, KVCacheV: [][]float32{{}}}
	if _, err := NewCPUDecodeStateFromMTPPromptContext(overflowModel, overflow, 1); err == nil {
		t.Fatal("accepted overflowing prompt-context KV width")
	}
}

func TestSeedCompressedKVFromPromptContext(t *testing.T) {
	m := newSingleLayerVerifierModel()
	ctx := MTPPromptContext{Tokens: []int{1}, PreviousToken: 1, Activation: []float32{0.5, 0.25}, SeqLen: 1, KVCacheK: [][]float32{{1, 2}}, KVCacheV: [][]float32{{3, 4}}}
	st, err := NewCPUDecodeStateFromMTPPromptContext(m, ctx, 3)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.seedCompressedKVFromPromptContext(st, ctx); err != nil {
		t.Fatal(err)
	}
	if st.CompressedKV == nil || len(st.CompressedKV) != 1 || st.CompressedKV[0] == nil || st.CompressedKV[0].SeqLen() != 1 {
		t.Fatalf("compressed state=%+v", st.CompressedKV)
	}
	if len(st.KVCacheK[0]) != 0 || len(st.KVCacheV[0]) != 0 {
		t.Fatalf("float KV retained after compressed seed K/V=%v/%v", st.KVCacheK[0], st.KVCacheV[0])
	}
	if !sameFloat32s(st.CompressedKV[0].GetK(), []float32{1, 2}) || !sameFloat32s(st.CompressedKV[0].GetV(), []float32{3, 4}) {
		t.Fatalf("compressed prompt K/V=%v/%v", st.CompressedKV[0].GetK(), st.CompressedKV[0].GetV())
	}
}

func TestGenerateMTPGraphFromPromptContextCompressedKV(t *testing.T) {
	m := newSingleLayerVerifierModel()
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	ctx := MTPPromptContext{Tokens: []int{1}, PreviousToken: 1, Activation: []float32{0.5, 0.25}, SeqLen: 1, KVCacheK: [][]float32{{1, 2}}, KVCacheV: [][]float32{{3, 4}}}
	res, err := m.GenerateMTPGraphFromPromptContext(d, ctx, nil, MTPGraphGenerationOptions{MaxTokens: 3, UseCompressedKV: true, Policy: MTPAdaptiveDraftPolicy{InitialDrafts: 2, MaxDrafts: 2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Steps) == 0 || res.GraphOutputTokens == 0 || len(res.Output) != len(ctx.Tokens)+3 {
		t.Fatalf("compressed MTP result steps=%v graph=%d output=%v", res.Steps, res.GraphOutputTokens, res.Output)
	}
}

func TestGenerateMTPGraphFromPromptContext(t *testing.T) {
	m := newZeroLayerVerifierModel()
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	ctx := MTPPromptContext{Tokens: []int{1}, PreviousToken: 1, Activation: []float32{0.5, 0.25}, SeqLen: 1, KVCacheK: [][]float32{}, KVCacheV: [][]float32{}}
	res, err := m.GenerateMTPGraphFromPromptContext(d, ctx, nil, MTPGraphGenerationOptions{MaxTokens: 3, Policy: MTPAdaptiveDraftPolicy{InitialDrafts: 2, MaxDrafts: 2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Steps) != 1 || len(res.StepSummaries) != 1 || res.GraphOutputTokens != len(res.Steps[0].OutputTokens) || res.GreedyTailTokens != 0 || res.Stats.Steps != 1 || res.Stats.DraftedTokens != 2 {
		t.Fatalf("result steps=%v stats=%+v", res.Steps, res.Stats)
	}
	if !sameInts(res.Output[:len(ctx.Tokens)], ctx.Tokens) {
		t.Fatalf("output prefix=%v want %v", res.Output, ctx.Tokens)
	}
	if len(res.Output) != len(ctx.Tokens)+len(res.Steps[0].OutputTokens) {
		t.Fatalf("output=%v step=%+v", res.Output, res.Steps[0])
	}
	if !sameInts(res.StepSummaries[0].OutputTokens, res.Steps[0].OutputTokens) || !sameInts(res.StepSummaries[0].Positions, res.Steps[0].Positions) || len(res.StepSummaries[0].DraftedTokens) != 2 || len(res.StepSummaries[0].VerifierTokens) != 3 {
		t.Fatalf("step summary=%+v commit=%+v", res.StepSummaries[0], res.Steps[0])
	}
	if res.HiddenSize != m.Config.HiddenSize || res.FinalState.PreviousToken < 0 || res.FinalState.PreviousToken >= m.Config.VocabSize || len(res.FinalState.Activation) != m.Config.HiddenSize || res.FinalStateOutputLen != len(res.Output) {
		t.Fatalf("hidden=%d final state=%+v covers=%d output=%v", res.HiddenSize, res.FinalState, res.FinalStateOutputLen, res.Output)
	}
	if !res.Capabilities.ExperimentalGenerationWiring || res.Capabilities.ReadyForPublicGeneration || !sameStringSet(res.MissingForPublicGeneration, res.Capabilities.MissingForPublicGeneration()) {
		t.Fatalf("capabilities=%+v missing=%v want %v", res.Capabilities, res.MissingForPublicGeneration, res.Capabilities.MissingForPublicGeneration())
	}
}

func TestMTPGraphGenerationResultValidate(t *testing.T) {
	caps := MTPGraphCapabilities{PublicGenerationWiring: false, VerifierBatchLayersGated: true, VerifierBatchLayers: false}
	valid := MTPGraphGenerationResult{
		Output:                     []int{9, 1, 2, 3},
		VocabSize:                  11,
		HiddenSize:                 2,
		RequestedMaxTokens:         3,
		Stats:                      MTPSpeculationStats{Steps: 1, DraftedTokens: 2, VerifiedTokens: 1, BonusTokens: 1, OutputTokens: 2},
		FinalState:                 MTPDrafterState{PreviousToken: 2, Activation: []float32{0.5, 0.25}},
		FinalStateOutputLen:        3,
		Steps:                      []MTPKVCommitPlan{{KeepTokens: 2, Positions: []int{1, 2}, OutputTokens: []int{1, 2}}},
		StepSummaries:              []MTPGraphGenerationStepSummary{{InputToken: 9, DraftedTokens: []int{1, 3}, VerifierTokens: []int{9, 1, 3}, VerifierOutputTokens: []int{1, 2, 3}, VerifierPositions: []int{1, 2, 3}, Positions: []int{1, 2}, AcceptedPrefixLen: 1, BonusToken: 2, OutputTokens: []int{1, 2}}},
		GraphOutputTokens:          2,
		GreedyTailTokens:           1,
		Capabilities:               caps,
		MissingForPublicGeneration: caps.MissingForPublicGeneration(),
	}
	if err := valid.Validate(1); err != nil {
		t.Fatalf("Validate valid: %v", err)
	}
	seeded := valid
	seeded.InitialStats = MTPSpeculationStats{Steps: 5, DraftedTokens: 7, VerifiedTokens: 3, BonusTokens: 5, OutputTokens: 11}
	seeded.Stats = MTPSpeculationStats{Steps: 6, DraftedTokens: 9, VerifiedTokens: 4, BonusTokens: 6, OutputTokens: 13}
	if err := seeded.Validate(1); err != nil {
		t.Fatalf("Validate seeded stats: %v", err)
	}
	bad := seeded
	bad.Stats.Steps = 4
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted stats moving backward from initial stats")
	}
	bad = valid
	bad.MissingForPublicGeneration = []string{"stale_blocker"}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted stale public-generation blocker list")
	}
	bad = valid
	bad.Capabilities.ReadyForPublicGeneration = true
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted stale capability readiness")
	}
	bad = valid
	bad.VocabSize = 0
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted missing vocab size")
	}
	bad = valid
	bad.VocabSize = 3
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted output token outside vocab")
	}
	bad = valid
	bad.HiddenSize = -1
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted negative hidden size")
	}
	bad = valid
	bad.HiddenSize = 0
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted missing hidden size")
	}
	bad = valid
	bad.StepSummaries[0].VerifierOutputTokens = []int{1, 11, 3}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted summary token outside vocab")
	}
	bad = valid
	bad.GraphOutputTokens = 1
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted wrong graph output total")
	}
	bad = valid
	bad.Output = []int{10, 1, 2, 3}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted cycle input token not matching output cursor")
	}
	bad = valid
	bad.Output = []int{9, 9, 9, 3}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted graph output content mismatch")
	}
	bad = valid
	bad.RequestedMaxTokens = -1
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted negative requested token budget")
	}
	bad = valid
	bad.RequestedMaxTokens = 0
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted generated tokens with zero requested budget")
	}
	bad = valid
	bad.RequestedMaxTokens = 2
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted requested token mismatch")
	}
	bad = valid
	bad.GreedyTailTokens = 0
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted wrong generated accounting")
	}
	bad = valid
	bad.Stats.OutputTokens = 99
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted stats/output mismatch")
	}
	bad = valid
	bad.FinalStateOutputLen = 0
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted missing final state cursor")
	}
	bad = valid
	bad.FinalStateOutputLen = 4
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted final state covering greedy tail")
	}
	bad = valid
	bad.FinalState.PreviousToken = 1
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted final state previous-token mismatch")
	}
	bad = valid
	bad.FinalState.Activation = []float32{1}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted final state activation width mismatch")
	}
	bad = valid
	bad.Stats.Steps = 99
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted stats/summary step mismatch")
	}
	bad = valid
	bad.Stats.DraftedTokens = 99
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted stats/summary drafted mismatch")
	}
	bad = valid
	bad.Stats.VerifiedTokens = 99
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted stats/summary verified mismatch")
	}
	bad = valid
	bad.Stats.BonusTokens = 99
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted stats/summary bonus mismatch")
	}
	bad = valid
	bad.Steps[0].KeepTokens = 3
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted malformed commit plan")
	}
	bad = valid
	bad.StepSummaries[0].OutputTokens = []int{9, 9}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted summary/commit mismatch")
	}
	bad = valid
	bad.StepSummaries[0].InputToken = -1
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted negative summary input token")
	}
	bad = valid
	bad.StepSummaries[0].DraftedTokens = []int{1, -2}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted negative drafted token")
	}
	bad = valid
	bad.StepSummaries[0].VerifierTokens = []int{8, 1, 2}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted summary verifier batch without input token")
	}
	bad = valid
	bad.Steps[0].OutputTokens = []int{2, 2}
	bad.StepSummaries[0].OutputTokens = []int{2, 2}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted summary output prefix not matching drafted prefix")
	}
	bad = valid
	bad.StepSummaries[0].VerifierTokens = []int{9, 1, 7}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted summary verifier/drafted mismatch")
	}
	bad = valid
	bad.StepSummaries[0].VerifierPositions = []int{1}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted short verifier positions")
	}
	bad = valid
	bad.StepSummaries[0].VerifierPositions = []int{2, 3, 4}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted verifier position not matching output cursor")
	}
	bad = valid
	bad.StepSummaries[0].VerifierPositions = []int{1, 3, 4}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted non-contiguous verifier positions")
	}
	bad = valid
	bad.StepSummaries[0].Positions = []int{2, 3}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted committed positions not matching verifier prefix")
	}
	bad = valid
	bad.StepSummaries[0].VerifierOutputTokens = []int{1}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted short verifier outputs")
	}
	bad = valid
	bad.StepSummaries[0].VerifierOutputTokens = []int{7, 2, 3}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted verifier output prefix not matching drafted prefix")
	}
	bad = valid
	bad.StepSummaries[0].VerifierOutputTokens = []int{1, 7, 3}
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted bonus not matching verifier output")
	}
	bad = valid
	bad.StepSummaries[0].VerifierOutputTokens = []int{1, 3, 3}
	bad.StepSummaries[0].AcceptedPrefixLen = 0
	bad.StepSummaries[0].OutputTokens = []int{1}
	bad.StepSummaries[0].BonusToken = 1
	bad.Steps[0].KeepTokens = 1
	bad.Steps[0].Positions = []int{1}
	bad.StepSummaries[0].VerifierPositions = []int{1, 2, 3}
	bad.Steps[0].OutputTokens = []int{1}
	bad.GraphOutputTokens = 1
	bad.GreedyTailTokens = 2
	bad.Stats.VerifiedTokens = 0
	bad.Stats.OutputTokens = 1
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted non-maximal accepted prefix")
	}
	bad = valid
	bad.StepSummaries[0].AllDraftsAccepted = true
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted inconsistent all-drafts-accepted=true")
	}
	bad = valid
	bad.StepSummaries[0].AcceptedPrefixLen = 2
	bad.StepSummaries[0].OutputTokens = []int{1, 2, 3}
	bad.StepSummaries[0].BonusToken = 3
	bad.Steps[0].KeepTokens = 3
	bad.Steps[0].Positions = []int{1, 2, 3}
	bad.Steps[0].OutputTokens = []int{1, 2, 3}
	bad.GraphOutputTokens = 3
	bad.GreedyTailTokens = 0
	bad.Stats.VerifiedTokens = 2
	bad.Stats.OutputTokens = 3
	if err := bad.Validate(1); err == nil {
		t.Fatal("accepted inconsistent all-drafts-accepted=false")
	}
	if err := valid.Validate(99); err == nil {
		t.Fatal("accepted bad prompt len")
	}
	zero := MTPGraphGenerationResult{Output: []int{10}, VocabSize: 11, HiddenSize: 2, InitialStats: MTPSpeculationStats{Steps: 2}, Stats: MTPSpeculationStats{Steps: 2}, FinalState: MTPDrafterState{PreviousToken: 10, Activation: []float32{1, 2}}, FinalStateOutputLen: 1, Capabilities: caps, MissingForPublicGeneration: caps.MissingForPublicGeneration()}
	if err := zero.Validate(1); err != nil {
		t.Fatalf("zero-cycle Validate: %v", err)
	}
	zero.Stats.Steps = 1
	if err := zero.Validate(1); err == nil {
		t.Fatal("accepted stale nonzero stats on zero-cycle result")
	}
	empty := MTPGraphGenerationResult{Capabilities: caps, MissingForPublicGeneration: []string{"stale"}}
	if err := empty.Validate(0); err == nil {
		t.Fatal("accepted stale public-generation blockers on empty output")
	}
}

func TestMTPExternalKVForDecodeStateRefreshesKVSlicesAndSeqLen(t *testing.T) {
	decode := &CPUDecodeState{Output: []int{1, 2, 3}, KVCacheK: [][]float32{{1, 2, 3, 4, 9, 10}}, KVCacheV: [][]float32{{5, 6, 7, 8, 11, 12}}}
	base := &MTPDrafterExternalKV{K: [][]float32{{1, 2}}, V: [][]float32{{3, 4}}, SourceLayers: []int{0}, SeqLen: 1}
	got, err := mtpExternalKVForDecodeState(decode, base)
	if err != nil {
		t.Fatal(err)
	}
	if got.SeqLen != len(decode.Output) || !sameFloat32s(got.K[0], decode.KVCacheK[0]) || !sameFloat32s(got.V[0], decode.KVCacheV[0]) || !sameInts(got.SourceLayers, base.SourceLayers) {
		t.Fatalf("refreshed external KV=%+v", got)
	}
	got.SourceLayers[0] = 99
	if base.SourceLayers[0] == 99 {
		t.Fatal("source layers alias base")
	}
	if nilKV, err := mtpExternalKVForDecodeState(decode, nil); err != nil || nilKV != nil {
		t.Fatalf("nil base got=%v err=%v", nilKV, err)
	}
	if _, err := mtpExternalKVForDecodeState(nil, base); err == nil {
		t.Fatal("accepted nil decode state")
	}
	badDecode := &CPUDecodeState{Output: []int{1}, KVCacheK: [][]float32{{}}, KVCacheV: nil}
	if _, err := mtpExternalKVForDecodeState(badDecode, base); err == nil {
		t.Fatal("accepted mismatched decode/base KV layers")
	}
}

func TestMTPExternalKVForDecodeStateRefreshesCompressedKVSlices(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	cache := kv.NewCompressedKVCache(2, 1, 2, nil, true)
	cache.Append([]float32{1, 2}, []float32{3, 4})
	cache.Append([]float32{5, 6}, []float32{7, 8})
	decode := &CPUDecodeState{Model: m, Output: []int{1, 2}, KVDims: []int{2}, CompressedKV: []*kv.CompressedKVCache{cache}}
	base := &MTPDrafterExternalKV{K: [][]float32{{1, 2}}, V: [][]float32{{3, 4}}, SourceLayers: []int{0}, SeqLen: 1}
	got, err := mtpExternalKVForDecodeState(decode, base)
	if err != nil {
		t.Fatal(err)
	}
	if got.SeqLen != 2 || !sameFloat32s(got.K[0], []float32{1, 2, 5, 6}) || !sameFloat32s(got.V[0], []float32{3, 4, 7, 8}) || !sameInts(got.SourceLayers, []int{0}) {
		t.Fatalf("compressed refreshed external KV=%+v", got)
	}
	got.K[0][0] = 99
	if cache.GetK()[0] == 99 {
		t.Fatal("refreshed compressed external KV aliases cache storage")
	}
}

func TestGenerateMTPGraphFromPromptContextGreedyDecodesSingleTokenTail(t *testing.T) {
	m := newZeroLayerVerifierModel()
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	ctx := MTPPromptContext{Tokens: []int{1}, PreviousToken: 1, Activation: []float32{0.5, 0.25}, SeqLen: 1, KVCacheK: [][]float32{}, KVCacheV: [][]float32{}}
	res, err := m.GenerateMTPGraphFromPromptContext(d, ctx, nil, MTPGraphGenerationOptions{MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Steps) != 0 || res.GraphOutputTokens != 0 || res.GreedyTailTokens != 1 || len(res.Output) != len(ctx.Tokens)+1 || !sameInts(res.Output[:len(ctx.Tokens)], ctx.Tokens) {
		t.Fatalf("single-tail result=%+v", res)
	}
	if res.HiddenSize != m.Config.HiddenSize || res.FinalStateOutputLen != len(ctx.Tokens) || res.FinalState.PreviousToken != ctx.PreviousToken || len(res.FinalState.Activation) != m.Config.HiddenSize {
		t.Fatalf("single-tail hidden=%d final state=%+v covers=%d, want prompt cursor %d", res.HiddenSize, res.FinalState, res.FinalStateOutputLen, len(ctx.Tokens))
	}
}
