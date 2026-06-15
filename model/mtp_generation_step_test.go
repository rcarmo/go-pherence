package model

import "testing"

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
	if !sameFloat32s(res.FinalState.Activation, res.Step.Verifier.FinalActivation) {
		t.Fatalf("final state activation=%v, want verifier activation %v", res.FinalState.Activation, res.Step.Verifier.FinalActivation)
	}
	res.Step.Verifier.FinalActivation[0] = 99
	if res.FinalState.Activation[0] == 99 {
		t.Fatal("final state aliases verifier final activation")
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
