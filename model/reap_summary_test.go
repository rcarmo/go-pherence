package model

import "testing"

func TestREAPSummary(t *testing.T) {
	reap := &REAPConfig{Enabled: true, PruneRatio: 0.2, Source: "filename_or_name", DefaultMask: map[int]bool{1: true, 3: true}, LayerActiveNumeric: map[int]map[int]bool{0: {2: true}, 1: {4: true, 5: true}}}
	s := reap.Summary()
	if !s.Enabled || s.PruneRatio != 0.2 || s.Source != "filename_or_name" || s.DefaultExperts != 2 || s.LayerMasks != 2 || s.LayerExpertTotal != 3 || !s.HasStaticMasks {
		t.Fatalf("bad summary: %+v", s)
	}
	if (REAPSummary{}) != (*REAPConfig)(nil).Summary() {
		t.Fatal("nil summary should be zero")
	}
}
