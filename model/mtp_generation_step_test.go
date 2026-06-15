package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestMTPAdaptiveDraftPolicyNextDraftCount(t *testing.T) {
	p := MTPAdaptiveDraftPolicy{MinDrafts: 1, InitialDrafts: 2, MaxDrafts: 4}
	if got := p.NextDraftCount(1, MTPSpeculationStats{}); got != 0 {
		t.Fatalf("remaining=1 draft=%d want 0", got)
	}
	if got := p.NextDraftCount(3, MTPSpeculationStats{}); got != 2 {
		t.Fatalf("cold draft=%d want 2", got)
	}
	high := MTPSpeculationStats{Steps: 2, DraftedTokens: 4, VerifiedTokens: 4}
	if got := p.NextDraftCount(8, high); got != 3 {
		t.Fatalf("high acceptance draft=%d want 3", got)
	}
	low := MTPSpeculationStats{Steps: 2, DraftedTokens: 4, VerifiedTokens: 0}
	if got := p.NextDraftCount(8, low); got != 1 {
		t.Fatalf("low acceptance draft=%d want 1", got)
	}
	if got := p.NextDraftCount(3, high); got != 2 { // remaining budget clamps G+1 output
		t.Fatalf("budget-clamped draft=%d want 2", got)
	}
	bad := MTPAdaptiveDraftPolicy{MinDrafts: 9, InitialDrafts: 99, MaxDrafts: 2, IncreaseAcceptance: 0.1, DecreaseAcceptance: 0.9}.Normalize()
	if bad.MinDrafts != 2 || bad.InitialDrafts != 2 || bad.MaxDrafts != 2 || bad.DecreaseAcceptance != bad.IncreaseAcceptance {
		t.Fatalf("Normalize bad policy=%+v", bad)
	}
}

func TestCPUDecodeStateRunMTPGraphDecodeStep(t *testing.T) {
	m := newZeroLayerVerifierModel()
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	st, err := NewCPUDecodeStateForSpeculative(m, []int{1}, 4)
	if err != nil {
		t.Fatal(err)
	}
	res, err := st.RunMTPGraphDecodeStep(d, state, nil, MTPGraphDecodeStepOptions{RemainingTokens: 3, DraftCount: 2}, MTPSpeculationStats{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Step.Drafts.Tokens) != 2 || res.Step.Graph.MaxKVKeepTokens != 3 || res.Commit.KeepTokens <= 0 {
		t.Fatalf("result drafts=%v graph=%+v commit=%+v", res.Step.Drafts.Tokens, res.Step.Graph, res.Commit)
	}
	if !sameInts(st.Output, append([]int{1}, res.Commit.OutputTokens...)) {
		t.Fatalf("decode output=%v commit=%v", st.Output, res.Commit.OutputTokens)
	}
	if res.Stats.Steps != 1 || res.Stats.DraftedTokens != 2 || res.Stats.BonusTokens != 1 {
		t.Fatalf("stats=%+v", res.Stats)
	}
	lastOutput := res.Commit.OutputTokens[len(res.Commit.OutputTokens)-1]
	if res.FinalState.PreviousToken != lastOutput {
		t.Fatalf("final state previous token=%d, want committed output %d", res.FinalState.PreviousToken, lastOutput)
	}
	committedActivation, err := res.Step.Verifier.CommittedActivation()
	if err != nil {
		t.Fatal(err)
	}
	if !sameFloat32s(res.FinalState.Activation, committedActivation) {
		t.Fatalf("final state activation=%v, want committed verifier activation %v", res.FinalState.Activation, committedActivation)
	}
	committedActivation[0] = 99
	if res.FinalState.Activation[0] == 99 {
		t.Fatal("final state aliases verifier committed activation")
	}
}

func TestMTPDrafterStateFromVerifierCommitUsesAcceptedRow(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{VocabSize: 4, HiddenSize: 2}}
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	verifier, err := NewMTPVerifierResultRowsForModel(m, 0, []int{1, 2}, [][]float32{
		{0, 0, 9, 0}, // mismatch at draft 0, bonus token 2
		{0, 9, 0, 0},
		{0, 0, 8, 0},
	}, [][]float32{
		{10, 11},
		{20, 21},
		{30, 31},
	})
	if err != nil {
		t.Fatal(err)
	}
	commit := MTPKVCommitPlan{KeepTokens: 1, Positions: []int{4}, OutputTokens: []int{2}}
	state, err := newMTPDrafterStateFromVerifierCommit(d, verifier, commit)
	if err != nil {
		t.Fatal(err)
	}
	if state.PreviousToken != 2 || !sameFloat32s(state.Activation, []float32{10, 11}) {
		t.Fatalf("state=%+v, want token 2 and accepted-row activation [10 11]", state)
	}
	verifier.ActivationRows[0][0] = 99
	if state.Activation[0] == 99 {
		t.Fatal("state aliases verifier activation row")
	}
}

func TestCPUDecodeStateRunMTPGraphDecodeStepCompressedKVZeroLayer(t *testing.T) {
	m := newZeroLayerVerifierModel()
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	st, err := NewCPUDecodeStateForSpeculative(m, []int{1}, 4)
	if err != nil {
		t.Fatal(err)
	}
	st.CompressedKV = []*kv.CompressedKVCache{}
	res, err := st.RunMTPGraphDecodeStep(d, state, nil, MTPGraphDecodeStepOptions{RemainingTokens: 3, DraftCount: 2}, MTPSpeculationStats{})
	if err != nil {
		t.Fatalf("RunMTPGraphDecodeStep compressed zero-layer: %v", err)
	}
	if !sameInts(st.Output, append([]int{1}, res.Commit.OutputTokens...)) {
		t.Fatalf("decode output=%v commit=%v", st.Output, res.Commit.OutputTokens)
	}
}

func TestCPUDecodeStateCompressedVerifierStagingValidation(t *testing.T) {
	m := &LlamaModel{Config: LlamaConfig{NumLayers: 1, NumKVHeads: 1, HeadDim: 2}, Layers: []LlamaLayer{{HasKV: true}}}
	st := &CPUDecodeState{Model: m, KVDims: []int{2}, CompressedKV: []*kv.CompressedKVCache{kv.NewCompressedKVCache(2, 1, 2, nil, true)}}
	st.CompressedKV[0].Append([]float32{1, 2}, []float32{3, 4})
	k, v, err := st.materializeCompressedKVForVerifier(1)
	if err != nil {
		t.Fatalf("materializeCompressedKVForVerifier: %v", err)
	}
	if !sameFloat32s(k[0], []float32{1, 2}) || !sameFloat32s(v[0], []float32{3, 4}) {
		t.Fatalf("materialized K/V=%v/%v", k[0], v[0])
	}
	k[0] = append(k[0], 5, 6, 7, 8)
	v[0] = append(v[0], 9, 10, 11, 12)
	if err := st.stageCompressedVerifierKV(k, v, 1); err != nil {
		t.Fatalf("stageCompressedVerifierKV: %v", err)
	}
	if got := st.CompressedKV[0].SeqLen(); got != 3 {
		t.Fatalf("compressed seq len=%d want 3", got)
	}
	badK := [][]float32{{1, 2, 3}}
	badV := [][]float32{{4, 5}}
	if err := st.stageCompressedVerifierKV(badK, badV, 3); err == nil {
		t.Fatal("accepted malformed staged compressed verifier K/V")
	}
}

func TestCPUDecodeStateRunMTPGraphDecodeStepRestoresOnCommitError(t *testing.T) {
	m := newZeroLayerVerifierModel()
	d := validProjectionOnlyDrafter()
	d.Config.VocabSize = m.Config.VocabSize
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	st, err := NewCPUDecodeStateForSpeculative(m, []int{1}, 4)
	if err != nil {
		t.Fatal(err)
	}
	_, err = st.RunMTPGraphDecodeStep(d, state, nil, MTPGraphDecodeStepOptions{RemainingTokens: 2, DraftCount: 2}, MTPSpeculationStats{})
	if err == nil {
		t.Fatal("accepted draft count that exceeds remaining budget")
	}
	if !sameInts(st.Output, []int{1}) {
		t.Fatalf("output mutated after failed step: %v", st.Output)
	}
}
