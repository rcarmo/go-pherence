package model

import "testing"

func TestNewMTPExecutionGraph(t *testing.T) {
	m := validDrafterStepBackboneModel()
	d := validDrafterStepScaffold()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	externalKV := &MTPDrafterExternalKV{K: [][]float32{{1, 0}}, V: [][]float32{{0, 1}}, SourceLayers: []int{0}, SeqLen: 1}
	graph, err := NewMTPExecutionGraph(m, d, state, externalKV, []int{2, 3, 1}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if graph.InputToken != 1 || graph.StartPos != 10 || graph.MaxKVKeepTokens != 4 {
		t.Fatalf("graph header=%+v", graph)
	}
	if !sameInts(graph.DraftedTokens, []int{2, 3, 1}) {
		t.Fatalf("drafted=%v", graph.DraftedTokens)
	}
	if !sameInts(graph.Verifier.VerifierTokens, []int{1, 2, 3, 1}) || !sameInts(graph.Verifier.Positions, []int{10, 11, 12, 13}) {
		t.Fatalf("verifier tokens=%v positions=%v", graph.Verifier.VerifierTokens, graph.Verifier.Positions)
	}
	wantInputs := []int{1, 2, 3}
	for i, step := range graph.DrafterSteps {
		if step.Index != i || step.InputToken != wantInputs[i] || step.ActivationWidth != d.BackboneHiddenSize || step.ExternalKVSeqLen != 1 || !sameInts(step.ExternalKVLayers, []int{0}) {
			t.Fatalf("step %d=%+v", i, step)
		}
	}
	graph.DrafterSteps[0].ExternalKVLayers[0] = 99
	if graph.DrafterSteps[1].ExternalKVLayers[0] != 0 {
		t.Fatalf("external KV layer slice aliases across steps: %+v", graph.DrafterSteps)
	}
}

func TestMTPExecutionGraphCommitPlan(t *testing.T) {
	m := validDrafterStepBackboneModel()
	d := validProjectionOnlyDrafter()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	graph, err := NewMTPExecutionGraph(m, d, state, nil, []int{2, 3, 1}, 20)
	if err != nil {
		t.Fatal(err)
	}
	acceptance, err := AcceptMTPDraft([]int{2, 3, 1}, []int{2, 8, 9, 10})
	if err != nil {
		t.Fatal(err)
	}
	commit, err := graph.CommitPlan(acceptance)
	if err != nil {
		t.Fatal(err)
	}
	if commit.KeepTokens != 2 || !sameInts(commit.Positions, []int{20, 21}) || !sameInts(commit.OutputTokens, []int{2, 8}) {
		t.Fatalf("commit=%+v", commit)
	}
	all, err := AcceptMTPDraft([]int{2, 3, 1}, []int{2, 3, 1, 7})
	if err != nil {
		t.Fatal(err)
	}
	commit, err = graph.CommitPlan(all)
	if err != nil {
		t.Fatal(err)
	}
	if commit.KeepTokens != 4 || !sameInts(commit.Positions, []int{20, 21, 22, 23}) || !sameInts(commit.OutputTokens, []int{2, 3, 1, 7}) {
		t.Fatalf("all accepted commit=%+v", commit)
	}
	wrong, err := AcceptMTPDraft([]int{2}, []int{2, 7})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := graph.CommitPlan(wrong); err == nil {
		t.Fatal("accepted mismatched acceptance draft count")
	}
}

func TestNewMTPExecutionGraphValidation(t *testing.T) {
	m := validDrafterStepBackboneModel()
	d := validProjectionOnlyDrafter()
	state, err := NewMTPDrafterState(1, []float32{0.5, 0.25}, d.BackboneHiddenSize)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := NewMTPExecutionGraph(nil, d, state, nil, nil, 0); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, err := NewMTPExecutionGraph(m, nil, state, nil, nil, 0); err == nil {
		t.Fatal("accepted nil drafter")
	}
	bad := state
	bad.Activation = []float32{1}
	if _, err := NewMTPExecutionGraph(m, d, bad, nil, nil, 0); err == nil {
		t.Fatal("accepted bad activation width")
	}
	if _, err := NewMTPExecutionGraph(m, d, state, nil, []int{m.Config.VocabSize}, 0); err == nil {
		t.Fatal("accepted drafted token outside vocab")
	}
	if _, err := NewMTPExecutionGraph(m, d, state, nil, nil, -1); err == nil {
		t.Fatal("accepted negative start position")
	}
}
