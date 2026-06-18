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
	if len(batch.HiddenFlat) != 2*m.Config.HiddenSize || len(batch.HiddenRows) != 2 || len(batch.HiddenRows[0]) != m.Config.HiddenSize || batch.HasPerLayerInputs {
		t.Fatalf("batch hiddenFlat=%v hidden=%v hasPLI=%v", batch.HiddenFlat, batch.HiddenRows, batch.HasPerLayerInputs)
	}
	batch.HiddenRows[0][0] = 123
	if batch.HiddenFlat[0] != 123 {
		t.Fatal("hidden row does not view flat backing buffer")
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
	if !batch.HasPerLayerInputs || len(batch.PerLayerInputFlat) != 4 || len(batch.PerLayerInputs) != 2 || len(batch.PerLayerInputs[0]) != 1 || len(batch.PerLayerInputs[0][0]) != 2 {
		t.Fatalf("PLI batch=%+v", batch)
	}
	first := append([]float32(nil), batch.PerLayerInputs[0][0]...)
	batch.PerLayerInputs[0][0][0] = 99
	if batch.PerLayerInputFlat[0] != 99 {
		t.Fatal("PLI row does not view flat backing buffer")
	}
	if batch.PerLayerInputs[1][0][0] == 99 {
		t.Fatalf("PLI rows alias across verifier tokens: %+v", batch.PerLayerInputs)
	}
	if len(first) != 2 || first[0] == batch.PerLayerInputs[1][0][0] {
		t.Fatalf("unexpected PLI rows first=%v second=%v", first, batch.PerLayerInputs[1][0])
	}
}

func TestMTPVerifierAttentionPlanMixedSlidingAndFullRanges(t *testing.T) {
	m := &LlamaModel{
		Config: LlamaConfig{VocabSize: 8, HiddenSize: 2, NumLayers: 2, SlidingWindow: 3, LayerTypes: []string{"sliding_attention", "full_attention"}},
		Layers: []LlamaLayer{{HasKV: true}, {HasKV: true}},
	}
	plan := MTPVerifierPlan{InputToken: 1, DraftedTokens: []int{2, 3}, VerifierTokens: []int{1, 2, 3}, StartPos: 4, Positions: []int{4, 5, 6}}
	attn, err := NewMTPVerifierAttentionPlan(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if attn.KVLen != 7 || !sameInts(attn.Positions, []int{4, 5, 6}) {
		t.Fatalf("attention plan header=%+v", attn)
	}
	if len(attn.Layers) != 2 {
		t.Fatalf("attention layers=%d", len(attn.Layers))
	}
	if !attn.Layers[0].Sliding || attn.Layers[0].SlidingWindow != 3 || !sameInts(attn.Layers[0].KVStart, []int{2, 3, 4}) || !sameInts(attn.Layers[0].KVEndExclusive, []int{5, 6, 7}) {
		t.Fatalf("sliding layer ranges=%+v", attn.Layers[0])
	}
	if attn.Layers[1].Sliding || attn.Layers[1].SlidingWindow != 0 || !sameInts(attn.Layers[1].KVStart, []int{0, 0, 0}) || !sameInts(attn.Layers[1].KVEndExclusive, []int{5, 6, 7}) {
		t.Fatalf("full layer ranges=%+v", attn.Layers[1])
	}
	bad := attn
	bad.Layers = append([]MTPVerifierLayerAttentionPlan(nil), attn.Layers...)
	bad.Layers[0].KVStart = append([]int(nil), attn.Layers[0].KVStart...)
	bad.Layers[0].KVStart[1]++
	if err := bad.ValidateAgainst(plan, m); err == nil {
		t.Fatal("accepted mutated verifier attention range")
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
	overflow := &LlamaModel{Config: LlamaConfig{VocabSize: 3, HiddenSize: int(^uint(0)>>1)/2 + 1, NumLayers: 0}, Layers: nil}
	overflowPlan := MTPVerifierPlan{InputToken: 1, DraftedTokens: []int{2}, VerifierTokens: []int{1, 2}, Positions: []int{0, 1}}
	if _, err := NewMTPVerifierBatchInputs(overflow, overflowPlan); err == nil {
		t.Fatal("accepted overflowing hidden batch size")
	}
}
