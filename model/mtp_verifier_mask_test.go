package model

import "testing"

func TestNewMTPVerifierAttentionPlanFullAndSliding(t *testing.T) {
	m := newZeroLayerVerifierModel()
	m.Config.NumLayers = 2
	m.Config.SlidingWindow = 3
	m.Config.LayerTypes = []string{"sliding_attention", "full_attention"}
	m.Layers = []LlamaLayer{{}, {}}
	plan := mustMTPVerifierPlan(t, m, 1, []int{2}, 4)
	mask, err := NewMTPVerifierAttentionPlan(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if mask.KVLen != 6 || !sameInts(mask.Positions, []int{4, 5}) || len(mask.Layers) != 2 {
		t.Fatalf("mask=%+v", mask)
	}
	sliding := mask.Layers[0]
	if !sliding.Sliding || sliding.SlidingWindow != 3 || !sameInts(sliding.KVStart, []int{2, 3}) || !sameInts(sliding.KVEndExclusive, []int{5, 6}) {
		t.Fatalf("sliding layer=%+v", sliding)
	}
	full := mask.Layers[1]
	if full.Sliding || full.SlidingWindow != 0 || !sameInts(full.KVStart, []int{0, 0}) || !sameInts(full.KVEndExclusive, []int{5, 6}) {
		t.Fatalf("full layer=%+v", full)
	}
	if err := mask.ValidateAgainst(plan, m); err != nil {
		t.Fatalf("ValidateAgainst: %v", err)
	}
}

func TestMTPVerifierBatchInputsCarriesAttentionPlan(t *testing.T) {
	m := newZeroLayerVerifierModel()
	m.Config.NumLayers = 1
	m.Config.SlidingWindow = 2
	m.Config.LayerTypes = []string{"sliding_attention"}
	m.Layers = []LlamaLayer{{}}
	plan := mustMTPVerifierPlan(t, m, 1, []int{2}, 2)
	batch, err := NewMTPVerifierBatchInputs(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	if err := batch.Attention.ValidateAgainst(batch.Plan, m); err != nil {
		t.Fatalf("batch attention invalid: %v", err)
	}
	if !sameInts(batch.Attention.Layers[0].KVStart, []int{1, 2}) || !sameInts(batch.Attention.Layers[0].KVEndExclusive, []int{3, 4}) {
		t.Fatalf("batch attention=%+v", batch.Attention.Layers[0])
	}
}

func TestMTPVerifierAttentionPlanValidation(t *testing.T) {
	m := newZeroLayerVerifierModel()
	m.Config.NumLayers = 1
	m.Layers = []LlamaLayer{{}}
	plan := mustMTPVerifierPlan(t, m, 1, []int{2}, 0)
	mask, err := NewMTPVerifierAttentionPlan(m, plan)
	if err != nil {
		t.Fatal(err)
	}
	bad := mask
	bad.KVLen = 99
	if err := bad.ValidateAgainst(plan, m); err == nil {
		t.Fatal("accepted bad KV len")
	}
	bad = mask
	bad.Layers = append(bad.Layers, MTPVerifierLayerAttentionPlan{})
	if err := bad.ValidateAgainst(plan, m); err == nil {
		t.Fatal("accepted bad layer count")
	}
	bad = cloneMTPVerifierAttentionPlan(mask)
	bad.Layers[0].KVStart[0] = -1
	if err := bad.ValidateAgainst(plan, m); err == nil {
		t.Fatal("accepted bad range")
	}

	mSliding := newZeroLayerVerifierModel()
	mSliding.Config.NumLayers = 1
	mSliding.Config.SlidingWindow = 3
	mSliding.Config.LayerTypes = []string{"sliding_attention"}
	mSliding.Layers = []LlamaLayer{{}}
	planSliding := mustMTPVerifierPlan(t, mSliding, 1, []int{2}, 4)
	maskSliding, err := NewMTPVerifierAttentionPlan(mSliding, planSliding)
	if err != nil {
		t.Fatal(err)
	}
	bad = cloneMTPVerifierAttentionPlan(maskSliding)
	bad.Layers[0].KVStart[0] = 0 // range-valid, but not the Gemma4 sliding-window start.
	if err := bad.ValidateAgainst(planSliding, mSliding); err == nil {
		t.Fatal("accepted wrong sliding attention range")
	}
	bad = cloneMTPVerifierAttentionPlan(maskSliding)
	bad.Layers[0].Sliding = false
	bad.Layers[0].SlidingWindow = 0
	if err := bad.ValidateAgainst(planSliding, mSliding); err == nil {
		t.Fatal("accepted wrong sliding attention metadata")
	}
}

func cloneMTPVerifierAttentionPlan(in MTPVerifierAttentionPlan) MTPVerifierAttentionPlan {
	out := MTPVerifierAttentionPlan{Positions: append([]int(nil), in.Positions...), KVLen: in.KVLen, Layers: make([]MTPVerifierLayerAttentionPlan, len(in.Layers))}
	for i, lp := range in.Layers {
		out.Layers[i] = MTPVerifierLayerAttentionPlan{
			Layer:          lp.Layer,
			Sliding:        lp.Sliding,
			SlidingWindow:  lp.SlidingWindow,
			KVStart:        append([]int(nil), lp.KVStart...),
			KVEndExclusive: append([]int(nil), lp.KVEndExclusive...),
		}
	}
	return out
}
