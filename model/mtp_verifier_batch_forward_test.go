package model

import "testing"

func TestRunMTPVerifierBatchForwardZeroLayer(t *testing.T) {
	m := newZeroLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 4)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.RunMTPVerifierBatchForward(batch, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	seq, err := m.RunMTPVerifierForward(plan, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !sameInts(got.VerifierTokens, seq.VerifierTokens) || !sameInts(got.Acceptance.OutputTokens, seq.Acceptance.OutputTokens) || !sameFloat32s(got.FinalActivation, seq.FinalActivation) {
		t.Fatalf("batch=%+v seq=%+v", got, seq)
	}
	if len(got.Logits) != len(seq.Logits) || !sameFloat32s(got.Logits[0], seq.Logits[0]) || !sameFloat32s(got.Logits[1], seq.Logits[1]) {
		t.Fatalf("batch logits=%v seq=%v", got.Logits, seq.Logits)
	}
}

func TestRunMTPVerifierBatchForwardLayeredSequentialLowering(t *testing.T) {
	m := newSingleLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	kvCacheK := make([][]float32, len(m.Layers))
	kvCacheV := make([][]float32, len(m.Layers))
	got, err := m.RunMTPVerifierBatchForward(batch, kvCacheK, kvCacheV)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Logits) != len(plan.VerifierTokens) || len(got.FinalActivation) != m.Config.HiddenSize {
		t.Fatalf("batch result logits=%d activation=%d", len(got.Logits), len(got.FinalActivation))
	}
	kvDim, err := m.LayerKVDim(0)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(kvCacheK[0]), len(plan.VerifierTokens)*kvDim; got != want {
		t.Fatalf("batch staged K len=%d want %d", got, want)
	}
}

func TestRunMTPVerifierBatchForwardValidation(t *testing.T) {
	m := newZeroLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	bad := batch
	bad.HiddenRows = bad.HiddenRows[:1]
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted short hidden rows")
	}
	bad = batch
	bad.HiddenRows = append([][]float32(nil), batch.HiddenRows...)
	bad.HiddenRows[0] = []float32{1}
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted wrong hidden width")
	}
	bad = batch
	bad.Attention.KVLen = 99
	if _, err := m.RunMTPVerifierBatchForward(bad, nil, nil); err == nil {
		t.Fatal("accepted bad attention plan")
	}
}
