package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/kv"
)

func layeredKVTestModel() *LlamaModel {
	return &LlamaModel{
		Config: LlamaConfig{NumLayers: 2, NumKVHeads: 1, HeadDim: 2, SlidingWindow: 2, LayerTypes: []string{"sliding_attention", "full_attention"}},
		Layers: []LlamaLayer{{HasKV: true}, {HasKV: true}},
	}
}

func appendLayeredRows(t *testing.T, s *CPUDecodeState, rows ...float32) {
	t.Helper()
	for _, x := range rows {
		for layer := 0; layer < 2; layer++ {
			v := x + float32(layer*100)
			if err := s.LayeredKV.Append(layer, []float32{v, v + .25}, []float32{-v, -v - .25}); err != nil {
				t.Fatal(err)
			}
		}
		s.Output = append(s.Output, int(x))
	}
}

func TestCPUDecodeStateEnableLayeredF32KVMixedLayers(t *testing.T) {
	m := layeredKVTestModel()
	s := &CPUDecodeState{Model: m, Output: []int{1, 2}, KVDims: []int{2, 2}, KVCacheK: [][]float32{{1, 1.25, 2, 2.25}, {101, 101.25, 102, 102.25}}, KVCacheV: [][]float32{{-1, -1.25, -2, -2.25}, {-101, -101.25, -102, -102.25}}}
	if err := s.EnableLayeredF32KV(1); err != nil {
		t.Fatal(err)
	}
	if s.LayeredKV == nil || len(s.KVCacheK[0]) != 0 || len(s.KVCacheV[1]) != 0 {
		t.Fatalf("migration failed: %+v", s)
	}
	view0, err := s.LayeredKV.View(0)
	if err != nil || view0.StartToken != 0 {
		t.Fatalf("ring view=%+v err=%v", view0, err)
	}
	view1, err := s.LayeredKV.View(1)
	if err != nil || view1.StartToken != 0 {
		t.Fatalf("full view=%+v err=%v", view1, err)
	}
	k, v, err := s.MaterializeLayeredKV()
	if err != nil {
		t.Fatal(err)
	}
	if !sameFloat32s(k[0], []float32{1, 1.25, 2, 2.25}) || !sameFloat32s(v[1], []float32{-101, -101.25, -102, -102.25}) {
		t.Fatalf("materialized K/V=%v/%v", k, v)
	}
}

func TestCPUDecodeStateLayeredCheckpointRestoreAndCommitAcrossWrap(t *testing.T) {
	m := layeredKVTestModel()
	s := &CPUDecodeState{Model: m, Output: []int{1, 2}, KVDims: []int{2, 2}, KVCacheK: [][]float32{{1, 1.25, 2, 2.25}, {101, 101.25, 102, 102.25}}, KVCacheV: [][]float32{{-1, -1.25, -2, -2.25}, {-101, -101.25, -102, -102.25}}}
	if err := s.EnableLayeredF32KV(1); err != nil {
		t.Fatal(err)
	}
	cp := s.Checkpoint()
	appendLayeredRows(t, s, 3, 4)
	if err := s.Restore(cp); err != nil {
		t.Fatal(err)
	}
	if len(s.Output) != 2 {
		t.Fatalf("output=%v", s.Output)
	}
	k, _, _, err := s.LayeredKV.MaterializeLayer(0)
	if err != nil {
		t.Fatal(err)
	}
	if !sameFloat32s(k, []float32{1, 1.25, 2, 2.25}) {
		t.Fatalf("restored ring=%v", k)
	}

	appendLayeredRows(t, s, 3, 4)
	acceptance := MTPAcceptance{DraftedCount: 1, VerifiedCount: 1, AcceptedPrefixLen: 1, AcceptedTokens: []int{3}, BonusToken: 4, OutputTokens: []int{3, 4}, AllDraftsAccepted: true, FirstRejectedIndex: -1}
	if err := s.CommitAccepted(cp, acceptance); err != nil {
		t.Fatal(err)
	}
	if !sameInts(s.Output, []int{1, 2, 3, 4}) {
		t.Fatalf("output=%v", s.Output)
	}
	k, _, start, err := s.LayeredKV.MaterializeLayer(0)
	if err != nil {
		t.Fatal(err)
	}
	if start != 1 || !sameFloat32s(k, []float32{2, 2.25, 3, 3.25, 4, 4.25}) {
		t.Fatalf("ring start=%d K=%v", start, k)
	}
	full, _, _, err := s.LayeredKV.MaterializeLayer(1)
	if err != nil {
		t.Fatal(err)
	}
	if len(full) != 8 {
		t.Fatalf("full len=%d", len(full))
	}
}

func TestCPUDecodeStateEnableLayeredF32KVRejectsMalformedSeed(t *testing.T) {
	m := layeredKVTestModel()
	bad := &CPUDecodeState{Model: m, Output: []int{1}, KVCacheK: [][]float32{{1, 2}, {1, 2}}, KVCacheV: [][]float32{{1}, {1, 2}}}
	if err := bad.EnableLayeredF32KV(1); err == nil {
		t.Fatal("accepted mismatched seed")
	}
	nonKV := &LlamaModel{Config: LlamaConfig{NumLayers: 1}, Layers: []LlamaLayer{{HasKV: false}}}
	bad = &CPUDecodeState{Model: nonKV, Output: []int{1}, KVCacheK: [][]float32{{1}}, KVCacheV: [][]float32{{1}}}
	if err := bad.EnableLayeredF32KV(1); err == nil {
		t.Fatal("accepted non-KV seed")
	}
}

func TestLayeredKVCheckpointTypePresent(t *testing.T) {
	var _ kv.LayeredF32KVCheckpoint = CPUDecodeCheckpoint{}.LayeredKV
}
