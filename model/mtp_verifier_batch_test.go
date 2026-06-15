package model

import "testing"

func TestNewMTPVerifierBatchInputsZeroLayer(t *testing.T) {
	m := newZeroLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 1, []int{2}, 5)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(batch.Plan.VerifierTokens, []int{1, 2}) || !sameInts(batch.Plan.Positions, []int{5, 6}) {
		t.Fatalf("batch plan=%+v", batch.Plan)
	}
	if len(batch.HiddenRows) != 2 || len(batch.HiddenRows[0]) != m.Config.HiddenSize || batch.HasPerLayerInputs {
		t.Fatalf("batch hidden=%v hasPLI=%v", batch.HiddenRows, batch.HasPerLayerInputs)
	}
	batch.Plan.VerifierTokens[0] = 99
	if plan.VerifierTokens[0] == 99 {
		t.Fatal("batch plan aliases source plan")
	}
}

func TestNewMTPVerifierBatchInputsWithPLI(t *testing.T) {
	m := newSingleLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	m.Config.HiddenPerLayer = 2
	m.PerLayerModelProj = []float32{1, 0, 0, 1}
	m.PerLayerProjNorm = []float32{1, 1}
	m.PerLayerProjScale = 1
	m.PerLayerInputScale = 1
	m.EmbedPerLayerScale = 1
	m.EmbedPerLayer = []float32{
		0, 0,
		10, 20,
		30, 40,
	}
	m.Config.VocabPerLayer = 3
	plan := mustMTPVerifierPlan(t, m, 1, []int{2}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if !batch.HasPerLayerInputs || len(batch.PerLayerInputs) != 2 || len(batch.PerLayerInputs[0]) != 1 || len(batch.PerLayerInputs[0][0]) != 2 {
		t.Fatalf("PLI batch=%+v", batch)
	}
	first := append([]float32(nil), batch.PerLayerInputs[0][0]...)
	batch.PerLayerInputs[0][0][0] = 99
	if batch.PerLayerInputs[1][0][0] == 99 {
		t.Fatalf("PLI rows alias across verifier tokens: %+v", batch.PerLayerInputs)
	}
	if len(first) != 2 || first[0] == batch.PerLayerInputs[1][0][0] {
		t.Fatalf("unexpected PLI rows first=%v second=%v", first, batch.PerLayerInputs[1][0])
	}
}

func TestNewMTPVerifierBatchInputsValidation(t *testing.T) {
	m := newZeroLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 1, []int{2}, 5)
	if _, err := NewMTPVerifierBatchInputs(nil, plan); err == nil {
		t.Fatal("accepted nil model")
	}
	bad := cloneMTPVerifierPlan(plan)
	bad.Positions[1] = 99
	if _, err := NewMTPVerifierBatchInputs(m, bad); err == nil {
		t.Fatal("accepted non-contiguous positions")
	}
	bad = cloneMTPVerifierPlan(plan)
	bad.VerifierTokens[1] = m.Config.VocabSize
	if _, err := NewMTPVerifierBatchInputs(m, bad); err == nil {
		t.Fatal("accepted out-of-vocab token")
	}
}
