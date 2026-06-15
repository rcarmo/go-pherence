package model

import "testing"

func TestMTPVerifierBatchScratchPlan(t *testing.T) {
	m := newSingleLayerVerifierModel()
	m.Config.SlidingWindow = 3
	m.Config.LayerTypes = []string{"sliding_attention"}
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 4)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	sp := batch.Scratch
	if sp.Batch != 2 || sp.HiddenSize != 2 || sp.MaxQDim != 2 || sp.MaxKVDim != 2 || sp.MaxIntermediate != 2 || sp.MaxAttentionRows != 6 || len(sp.Layers) != 1 {
		t.Fatalf("scratch=%+v", sp)
	}
	lp := sp.Layers[0]
	if lp.Layer != 0 || lp.Batch != 2 || lp.QDim != 2 || lp.KVDim != 2 || lp.Intermediate != 2 || !lp.HasKV || lp.SharedKV || lp.AttentionRows != 6 {
		t.Fatalf("layer scratch=%+v", lp)
	}
	if lp.AttentionOutFloats != 4 || lp.AttentionScoreFloats != 12 || lp.MLPFloats != 4 || lp.TotalFloat32 != 20 || sp.TotalFloat32 != 20 {
		t.Fatalf("layer totals=%+v plan=%+v", lp, sp)
	}
}

func TestMTPVerifierBatchScratchPlanWithPLI(t *testing.T) {
	m := newSingleLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	m.Config.HiddenPerLayer = 2
	m.PerLayerModelProj = []float32{1, 0, 0, 1}
	m.PerLayerProjNorm = []float32{1, 1}
	m.PerLayerProjScale = 1
	m.PerLayerInputScale = 1
	m.EmbedPerLayerScale = 1
	m.Layers[0].PLIGate = []float32{1, 0, 0, 1}
	m.Layers[0].PLIProj = []float32{1, 0, 0, 1}
	m.Layers[0].PLIPostNorm = []float32{1, 1}
	plan := mustMTPVerifierPlan(t, m, 0, []int{1}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if !batch.HasPerLayerInputs || batch.Scratch.Layers[0].PLIFloats != 4 || batch.Scratch.Layers[0].TotalFloat32 != 16 {
		t.Fatalf("PLI scratch=%+v hasPLI=%v", batch.Scratch.Layers[0], batch.HasPerLayerInputs)
	}
}

func TestMTPVerifierBatchScratchPlanValidation(t *testing.T) {
	m := newZeroLayerVerifierModel()
	plan := mustMTPVerifierPlan(t, m, 1, []int{2}, 0)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	bad := batch
	bad.HiddenRows = bad.HiddenRows[:1]
	if _, err := NewMTPVerifierBatchScratchPlan(m, bad); err == nil {
		t.Fatal("accepted batch with short hidden rows")
	}
	if _, err := NewMTPVerifierBatchScratchPlan(nil, batch); err == nil {
		t.Fatal("accepted nil model")
	}
}
