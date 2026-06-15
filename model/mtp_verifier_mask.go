package model

import "fmt"

// MTPVerifierAttentionPlan describes the verifier-side attention window for the
// materialized [input]+drafted batch. It is represented as per-row KV ranges
// rather than a dense bool mask so CPU, SIMD, and GPU backends can lower it into
// their native graph form.
type MTPVerifierAttentionPlan struct {
	Positions []int
	KVLen     int
	Layers    []MTPVerifierLayerAttentionPlan
}

type MTPVerifierLayerAttentionPlan struct {
	Layer          int
	Sliding        bool
	SlidingWindow  int
	KVStart        []int
	KVEndExclusive []int
}

func NewMTPVerifierAttentionPlan(m *LlamaModel, plan MTPVerifierPlan) (MTPVerifierAttentionPlan, error) {
	if err := validateMTPVerifierPlanForModel(m, plan); err != nil {
		return MTPVerifierAttentionPlan{}, err
	}
	if len(plan.Positions) == 0 {
		return MTPVerifierAttentionPlan{}, fmt.Errorf("empty verifier positions")
	}
	kvLen := plan.Positions[len(plan.Positions)-1] + 1
	layers := make([]MTPVerifierLayerAttentionPlan, m.Config.NumLayers)
	for l := 0; l < m.Config.NumLayers; l++ {
		sliding := false
		window := 0
		if m.Config.SlidingWindow > 0 && len(m.Config.LayerTypes) > l && m.Config.LayerTypes[l] == "sliding_attention" {
			sliding = true
			window = m.Config.SlidingWindow
		}
		starts := make([]int, len(plan.Positions))
		ends := make([]int, len(plan.Positions))
		for i, pos := range plan.Positions {
			end := pos + 1
			start := 0
			if sliding && end > window {
				start = end - window
			}
			starts[i] = start
			ends[i] = end
		}
		layers[l] = MTPVerifierLayerAttentionPlan{Layer: l, Sliding: sliding, SlidingWindow: window, KVStart: starts, KVEndExclusive: ends}
	}
	return MTPVerifierAttentionPlan{Positions: append([]int(nil), plan.Positions...), KVLen: kvLen, Layers: layers}, nil
}

func (p MTPVerifierAttentionPlan) ValidateAgainst(plan MTPVerifierPlan, m *LlamaModel) error {
	if err := validateMTPVerifierPlanForModel(m, plan); err != nil {
		return err
	}
	if !mtpSameInts(p.Positions, plan.Positions) {
		return fmt.Errorf("MTP verifier mask positions=%v, want %v", p.Positions, plan.Positions)
	}
	wantKVLen := plan.Positions[len(plan.Positions)-1] + 1
	if p.KVLen != wantKVLen {
		return fmt.Errorf("MTP verifier mask kvLen=%d, want %d", p.KVLen, wantKVLen)
	}
	if len(p.Layers) != m.Config.NumLayers {
		return fmt.Errorf("MTP verifier mask layers=%d, want %d", len(p.Layers), m.Config.NumLayers)
	}
	for l, lp := range p.Layers {
		if lp.Layer != l || len(lp.KVStart) != len(plan.Positions) || len(lp.KVEndExclusive) != len(plan.Positions) {
			return fmt.Errorf("MTP verifier mask malformed layer=%d plan=%+v", l, lp)
		}
		for i := range plan.Positions {
			if lp.KVStart[i] < 0 || lp.KVStart[i] > lp.KVEndExclusive[i] || lp.KVEndExclusive[i] > p.KVLen {
				return fmt.Errorf("MTP verifier mask invalid layer=%d row=%d range=[%d,%d) kvLen=%d", l, i, lp.KVStart[i], lp.KVEndExclusive[i], p.KVLen)
			}
		}
	}
	return nil
}
