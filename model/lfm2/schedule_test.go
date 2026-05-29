package lfm2

import (
	"path/filepath"
	"testing"
)

func TestLayerScheduleFromFixture(t *testing.T) {
	meta, err := LoadReferenceMetadata(filepath.Join("testdata", "lfm25_8b_a1b_metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	s, err := NewLayerSchedule(meta.Config)
	if err != nil {
		t.Fatal(err)
	}
	if len(s.Steps) != 24 || len(s.ConvIndices) != 21 || len(s.FullAttentionIndices) != 3 {
		t.Fatalf("schedule=%+v", s)
	}
	wantAttn := []int{7, 15, 23}
	for i, want := range wantAttn {
		if s.FullAttentionIndices[i] != want || !s.IsFullAttentionLayer(want) || s.IsConvLayer(want) {
			t.Fatalf("attention indices=%v", s.FullAttentionIndices)
		}
	}
	for _, idx := range []int{0, 1, 22} {
		if !s.IsConvLayer(idx) || s.IsFullAttentionLayer(idx) {
			t.Fatalf("conv membership failed for %d", idx)
		}
	}
}

func TestLayerScheduleRejectsMalformed(t *testing.T) {
	s := LayerSchedule{Steps: []LayerStep{{Index: 1, Kind: LayerConv}}, ConvIndices: []int{1}}
	if err := s.Validate(1); err == nil {
		t.Fatal("expected non-sequential index error")
	}
	s = LayerSchedule{Steps: []LayerStep{{Index: 0, Kind: "bad"}}}
	if err := s.Validate(1); err == nil {
		t.Fatal("expected bad kind error")
	}
}
