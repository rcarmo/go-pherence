package kv

import "testing"

func TestLayeredF32KVMixedSlidingAndFull(t *testing.T) {
	cache, err := NewLayeredF32KV([]LayerF32KVConfig{
		{Dim: 2, Sliding: true, SlidingWindow: 3},
		{Dim: 2},
	}, 2)
	if err != nil {
		t.Fatalf("NewLayeredF32KV: %v", err)
	}
	for i := 1; i <= 6; i++ {
		k := []float32{float32(i), float32(i + 100)}
		v := []float32{float32(i + 1000), float32(i + 2000)}
		if err := cache.Append(0, k, v); err != nil {
			t.Fatalf("Append sliding #%d: %v", i, err)
		}
		if err := cache.Append(1, k, v); err != nil {
			t.Fatalf("Append full #%d: %v", i, err)
		}
	}
	if err := cache.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	view0, err := cache.View(0)
	if err != nil {
		t.Fatalf("View(0): %v", err)
	}
	if view0.StartToken != 1 {
		t.Fatalf("View(0).StartToken=%d want 1", view0.StartToken)
	}
	if got, want := view0.FirstK, []float32{2, 102, 3, 103, 4, 104, 5, 105}; !sameFloat32s(got, want) {
		t.Fatalf("View(0).FirstK=%v want %v", got, want)
	}
	if got, want := view0.FirstV, []float32{1002, 2002, 1003, 2003, 1004, 2004, 1005, 2005}; !sameFloat32s(got, want) {
		t.Fatalf("View(0).FirstV=%v want %v", got, want)
	}
	if got, want := view0.SecondK, []float32{6, 106}; !sameFloat32s(got, want) {
		t.Fatalf("View(0).SecondK=%v want %v", got, want)
	}
	if got, want := view0.SecondV, []float32{1006, 2006}; !sameFloat32s(got, want) {
		t.Fatalf("View(0).SecondV=%v want %v", got, want)
	}

	k0, v0, start0, err := cache.MaterializeLayer(0)
	if err != nil {
		t.Fatalf("MaterializeLayer(0): %v", err)
	}
	if start0 != 1 {
		t.Fatalf("MaterializeLayer(0).StartToken=%d want 1", start0)
	}
	if got, want := k0, []float32{2, 102, 3, 103, 4, 104, 5, 105, 6, 106}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(0).K=%v want %v", got, want)
	}
	if got, want := v0, []float32{1002, 2002, 1003, 2003, 1004, 2004, 1005, 2005, 1006, 2006}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(0).V=%v want %v", got, want)
	}

	view1, err := cache.View(1)
	if err != nil {
		t.Fatalf("View(1): %v", err)
	}
	if view1.StartToken != 0 {
		t.Fatalf("View(1).StartToken=%d want 0", view1.StartToken)
	}
	if got, want := view1.FirstK, []float32{1, 101, 2, 102, 3, 103, 4, 104, 5, 105, 6, 106}; !sameFloat32s(got, want) {
		t.Fatalf("View(1).FirstK=%v want %v", got, want)
	}
	if len(view1.SecondK) != 0 || len(view1.SecondV) != 0 {
		t.Fatalf("View(1) unexpectedly wrapped: %+v", view1)
	}

	k1, v1, start1, err := cache.MaterializeLayer(1)
	if err != nil {
		t.Fatalf("MaterializeLayer(1): %v", err)
	}
	if start1 != 0 {
		t.Fatalf("MaterializeLayer(1).StartToken=%d want 0", start1)
	}
	if got, want := k1, []float32{1, 101, 2, 102, 3, 103, 4, 104, 5, 105, 6, 106}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(1).K=%v want %v", got, want)
	}
	if got, want := v1, []float32{1001, 2001, 1002, 2002, 1003, 2003, 1004, 2004, 1005, 2005, 1006, 2006}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(1).V=%v want %v", got, want)
	}

	if got, want := cache.Bytes(), int64((5*2+6*2)*8); got != want {
		t.Fatalf("Bytes=%d want %d", got, want)
	}

	cache.Reset()
	for layer := 0; layer < 2; layer++ {
		view, err := cache.View(layer)
		if err != nil {
			t.Fatalf("View(%d) after Reset: %v", layer, err)
		}
		if view.StartToken != 0 || len(view.FirstK) != 0 || len(view.FirstV) != 0 || len(view.SecondK) != 0 || len(view.SecondV) != 0 {
			t.Fatalf("View(%d) after Reset=%+v", layer, view)
		}
	}
}

func TestLayeredF32KVMultipleWraps(t *testing.T) {
	cache, err := NewLayeredF32KV([]LayerF32KVConfig{{Dim: 1, Sliding: true, SlidingWindow: 2}}, 1)
	if err != nil {
		t.Fatalf("NewLayeredF32KV: %v", err)
	}
	for i := 1; i <= 8; i++ {
		if err := cache.Append(0, []float32{float32(i)}, []float32{float32(i + 100)}); err != nil {
			t.Fatalf("Append #%d: %v", i, err)
		}
	}
	view, err := cache.View(0)
	if err != nil {
		t.Fatalf("View(0): %v", err)
	}
	if view.StartToken != 5 {
		t.Fatalf("View(0).StartToken=%d want 5", view.StartToken)
	}
	if got, want := view.FirstK, []float32{6}; !sameFloat32s(got, want) {
		t.Fatalf("View(0).FirstK=%v want %v", got, want)
	}
	if got, want := view.SecondK, []float32{7, 8}; !sameFloat32s(got, want) {
		t.Fatalf("View(0).SecondK=%v want %v", got, want)
	}
	k, v, startToken, err := cache.MaterializeLayer(0)
	if err != nil {
		t.Fatalf("MaterializeLayer(0): %v", err)
	}
	if startToken != 5 {
		t.Fatalf("MaterializeLayer(0).StartToken=%d want 5", startToken)
	}
	if got, want := k, []float32{6, 7, 8}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(0).K=%v want %v", got, want)
	}
	if got, want := v, []float32{106, 107, 108}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(0).V=%v want %v", got, want)
	}
}

func TestLayeredF32KVExactFullLayer(t *testing.T) {
	cache, err := NewLayeredF32KV([]LayerF32KVConfig{{Dim: 1}}, 4)
	if err != nil {
		t.Fatalf("NewLayeredF32KV: %v", err)
	}
	for i := 1; i <= 4; i++ {
		if err := cache.Append(0, []float32{float32(i)}, []float32{float32(i + 10)}); err != nil {
			t.Fatalf("Append #%d: %v", i, err)
		}
	}
	k, v, startToken, err := cache.MaterializeLayer(0)
	if err != nil {
		t.Fatalf("MaterializeLayer(0): %v", err)
	}
	if startToken != 0 {
		t.Fatalf("MaterializeLayer(0).StartToken=%d want 0", startToken)
	}
	if got, want := k, []float32{1, 2, 3, 4}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(0).K=%v want %v", got, want)
	}
	if got, want := v, []float32{11, 12, 13, 14}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(0).V=%v want %v", got, want)
	}
}

func TestLayeredF32KVCapacitySizing(t *testing.T) {
	cache, err := NewLayeredF32KV([]LayerF32KVConfig{
		{Dim: 4, Sliding: true, SlidingWindow: 64},
		{Dim: 8},
		{Dim: 2, Sliding: true, SlidingWindow: 1},
	}, 16)
	if err != nil {
		t.Fatalf("NewLayeredF32KV: %v", err)
	}
	if err := cache.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if got := len(cache.stores); got != 3 {
		t.Fatalf("stores=%d want 3", got)
	}
	if ring, ok := cache.stores[0].(*RingF32KV); !ok {
		t.Fatalf("layer 0 type=%T want *RingF32KV", cache.stores[0])
	} else if got, want := ring.Capacity(), 80; got != want {
		t.Fatalf("layer 0 capacity=%d want %d", got, want)
	}
	if _, ok := cache.stores[1].(*LinearF32KV); !ok {
		t.Fatalf("layer 1 type=%T want *LinearF32KV", cache.stores[1])
	}
	if ring, ok := cache.stores[2].(*RingF32KV); !ok {
		t.Fatalf("layer 2 type=%T want *RingF32KV", cache.stores[2])
	} else if got, want := ring.Capacity(), 17; got != want {
		t.Fatalf("layer 2 capacity=%d want %d", got, want)
	}
}

func TestLayeredF32KVCheckpointRestoreAfterWrap(t *testing.T) {
	configs := []LayerF32KVConfig{{Dim: 1, Sliding: true, SlidingWindow: 2}, {Dim: 1}}
	cache, err := NewLayeredF32KV(configs, 1)
	if err != nil {
		t.Fatalf("NewLayeredF32KV source: %v", err)
	}
	for i := 1; i <= 5; i++ {
		if err := cache.Append(0, []float32{float32(i)}, []float32{float32(i + 100)}); err != nil {
			t.Fatalf("Append sliding #%d: %v", i, err)
		}
		if err := cache.Append(1, []float32{float32(i)}, []float32{float32(i + 200)}); err != nil {
			t.Fatalf("Append full #%d: %v", i, err)
		}
	}
	cp := cache.Checkpoint()
	if !cp.valid {
		t.Fatal("Checkpoint returned invalid checkpoint")
	}

	restored, err := NewLayeredF32KV(configs, 1)
	if err != nil {
		t.Fatalf("NewLayeredF32KV restored: %v", err)
	}
	if err := restored.Restore(cp); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if err := restored.Validate(); err != nil {
		t.Fatalf("Validate after Restore: %v", err)
	}

	k0, v0, start0, err := restored.MaterializeLayer(0)
	if err != nil {
		t.Fatalf("MaterializeLayer(0): %v", err)
	}
	if start0 != 2 {
		t.Fatalf("MaterializeLayer(0).StartToken=%d want 2", start0)
	}
	if got, want := k0, []float32{3, 4, 5}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(0).K=%v want %v", got, want)
	}
	if got, want := v0, []float32{103, 104, 105}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(0).V=%v want %v", got, want)
	}

	k1, v1, start1, err := restored.MaterializeLayer(1)
	if err != nil {
		t.Fatalf("MaterializeLayer(1): %v", err)
	}
	if start1 != 0 {
		t.Fatalf("MaterializeLayer(1).StartToken=%d want 0", start1)
	}
	if got, want := k1, []float32{1, 2, 3, 4, 5}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(1).K=%v want %v", got, want)
	}
	if got, want := v1, []float32{201, 202, 203, 204, 205}; !sameFloat32s(got, want) {
		t.Fatalf("MaterializeLayer(1).V=%v want %v", got, want)
	}

	if err := restored.Append(0, []float32{6}, []float32{106}); err != nil {
		t.Fatalf("Append restored sliding: %v", err)
	}
	if err := restored.Append(1, []float32{6}, []float32{206}); err != nil {
		t.Fatalf("Append restored full: %v", err)
	}
	k0, _, start0, err = restored.MaterializeLayer(0)
	if err != nil {
		t.Fatalf("MaterializeLayer(0) after Append: %v", err)
	}
	if start0 != 3 || !sameFloat32s(k0, []float32{4, 5, 6}) {
		t.Fatalf("post-restore wrap mismatch start=%d K=%v", start0, k0)
	}
}

func TestLayeredF32KVKeepAppendedMixedWrapKeepVariants(t *testing.T) {
	configs := []LayerF32KVConfig{{Dim: 1, Sliding: true, SlidingWindow: 2}, {Dim: 1}}
	tests := []struct {
		name       string
		keepTokens int
		wantStart0 int
		wantK0     []float32
		wantV0     []float32
		wantStart1 int
		wantK1     []float32
		wantV1     []float32
	}{
		{
			name:       "keep none",
			keepTokens: 0,
			wantStart0: 0,
			wantK0:     []float32{1, 2, 3},
			wantV0:     []float32{101, 102, 103},
			wantStart1: 0,
			wantK1:     []float32{1, 2, 3},
			wantV1:     []float32{101, 102, 103},
		},
		{
			name:       "keep partial",
			keepTokens: 2,
			wantStart0: 1,
			wantK0:     []float32{2, 3, 4, 5},
			wantV0:     []float32{102, 103, 104, 105},
			wantStart1: 0,
			wantK1:     []float32{1, 2, 3, 4, 5},
			wantV1:     []float32{101, 102, 103, 104, 105},
		},
		{
			name:       "keep all",
			keepTokens: 3,
			wantStart0: 2,
			wantK0:     []float32{3, 4, 5, 6},
			wantV0:     []float32{103, 104, 105, 106},
			wantStart1: 0,
			wantK1:     []float32{1, 2, 3, 4, 5, 6},
			wantV1:     []float32{101, 102, 103, 104, 105, 106},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache, err := NewLayeredF32KV(configs, 2)
			if err != nil {
				t.Fatalf("NewLayeredF32KV: %v", err)
			}
			for token := 1; token <= 3; token++ {
				appendLayeredTokenAllLayers(t, cache, token)
			}
			cp := cache.Checkpoint()
			if !cp.valid {
				t.Fatal("Checkpoint returned invalid checkpoint")
			}
			for token := 4; token <= 6; token++ {
				appendLayeredTokenAllLayers(t, cache, token)
			}
			requireLayeredMaterialized(t, cache, 0, 2, []float32{3, 4, 5, 6}, []float32{103, 104, 105, 106})
			if err := cache.KeepAppended(cp, tt.keepTokens); err != nil {
				t.Fatalf("KeepAppended(%d): %v", tt.keepTokens, err)
			}
			requireLayeredMaterialized(t, cache, 0, tt.wantStart0, tt.wantK0, tt.wantV0)
			requireLayeredMaterialized(t, cache, 1, tt.wantStart1, tt.wantK1, tt.wantV1)
			if err := cache.Validate(); err != nil {
				t.Fatalf("Validate after KeepAppended: %v", err)
			}
		})
	}
}

func TestLayeredF32KVKeepAppendedRejectsEvictedKeptPrefixAtomically(t *testing.T) {
	cache, err := NewLayeredF32KV([]LayerF32KVConfig{{Dim: 1, Sliding: true, SlidingWindow: 2}, {Dim: 1}}, 1)
	if err != nil {
		t.Fatalf("NewLayeredF32KV: %v", err)
	}
	appendLayeredTokenAllLayers(t, cache, 1)
	cp := cache.Checkpoint()
	if !cp.valid {
		t.Fatal("Checkpoint returned invalid checkpoint")
	}
	for token := 2; token <= 5; token++ {
		appendLayeredTokenAllLayers(t, cache, token)
	}
	before := snapshotLayeredF32KVState(t, cache)
	requireLayeredMaterialized(t, cache, 0, 2, []float32{3, 4, 5}, []float32{103, 104, 105})
	if err := cache.KeepAppended(cp, 1); err == nil {
		t.Fatal("KeepAppended accepted unrecoverable staged prefix")
	}
	requireLayeredSnapshot(t, cache, before)
}

func TestLayeredF32KVKeepAppendedRejectsMismatchedStagingAtomically(t *testing.T) {
	cache, err := NewLayeredF32KV([]LayerF32KVConfig{{Dim: 1}, {Dim: 1}}, 1)
	if err != nil {
		t.Fatalf("NewLayeredF32KV: %v", err)
	}
	for token := 1; token <= 2; token++ {
		appendLayeredTokenAllLayers(t, cache, token)
	}
	cp := cache.Checkpoint()
	if !cp.valid {
		t.Fatal("Checkpoint returned invalid checkpoint")
	}
	appendLayeredToken(t, cache, 0, 3)
	appendLayeredToken(t, cache, 1, 3)
	appendLayeredToken(t, cache, 1, 4)
	before := snapshotLayeredF32KVState(t, cache)
	if err := cache.KeepAppended(cp, 1); err == nil {
		t.Fatal("KeepAppended accepted mismatched staged counts")
	}
	requireLayeredSnapshot(t, cache, before)
}

func TestLayeredF32KVMalformedAndOverflow(t *testing.T) {
	if _, err := NewLayeredF32KV([]LayerF32KVConfig{{Dim: 0}}, 1); err == nil {
		t.Fatal("NewLayeredF32KV accepted dim=0")
	}
	if _, err := NewLayeredF32KV([]LayerF32KVConfig{{Dim: 1, Sliding: true, SlidingWindow: 0}}, 1); err == nil {
		t.Fatal("NewLayeredF32KV accepted slidingWindow=0")
	}
	if _, err := NewLayeredF32KV([]LayerF32KVConfig{{Dim: 1, SlidingWindow: 1}}, 1); err == nil {
		t.Fatal("NewLayeredF32KV accepted non-sliding layer with sliding window")
	}
	maxInt := int(^uint(0) >> 1)
	if _, err := NewLayeredF32KV([]LayerF32KVConfig{{Dim: 1, Sliding: true, SlidingWindow: maxInt}}, 1); err == nil {
		t.Fatal("NewLayeredF32KV accepted capacity overflow")
	}
	if _, _, err := EstimateLayeredF32KVBytes([]LayerF32KVConfig{{Dim: 1, Sliding: true, SlidingWindow: maxInt}}, 1, 1); err == nil {
		t.Fatal("EstimateLayeredF32KVBytes accepted capacity overflow")
	}
	if _, _, err := EstimateLayeredF32KVBytes([]LayerF32KVConfig{{Dim: 1}}, -1, 1); err == nil {
		t.Fatal("EstimateLayeredF32KVBytes accepted negative maxContext")
	}

	cache, err := NewLayeredF32KV([]LayerF32KVConfig{{Dim: 1}}, 1)
	if err != nil {
		t.Fatalf("NewLayeredF32KV: %v", err)
	}
	if err := cache.Append(1, []float32{1}, []float32{2}); err == nil {
		t.Fatal("Append accepted out-of-range layer")
	}
	if _, err := cache.View(-1); err == nil {
		t.Fatal("View accepted negative layer")
	}
	if _, _, _, err := cache.MaterializeLayer(2); err == nil {
		t.Fatal("MaterializeLayer accepted out-of-range layer")
	}
	if err := cache.Restore(LayeredF32KVCheckpoint{}); err == nil {
		t.Fatal("Restore accepted invalid layered checkpoint")
	}
	if err := cache.KeepAppended(LayeredF32KVCheckpoint{}, 0); err == nil {
		t.Fatal("KeepAppended accepted invalid layered checkpoint")
	}
	cp := cache.Checkpoint()
	if err := cache.KeepAppended(cp, -1); err == nil {
		t.Fatal("KeepAppended accepted negative keepTokens")
	}
	cp.maxPrefillChunk++
	if err := cache.Restore(cp); err == nil {
		t.Fatal("Restore accepted mismatched maxPrefillChunk")
	}
	cp = cache.Checkpoint()
	cp.configs[0].Dim = 2
	if err := cache.Restore(cp); err == nil {
		t.Fatal("Restore accepted mismatched config")
	}
}

type layeredF32KVState struct {
	startToken int
	k          []float32
	v          []float32
}

func appendLayeredTokenAllLayers(t *testing.T, cache *LayeredF32KV, token int) {
	t.Helper()
	for layer := range cache.stores {
		appendLayeredToken(t, cache, layer, token)
	}
}

func appendLayeredToken(t *testing.T, cache *LayeredF32KV, layer, token int) {
	t.Helper()
	if err := cache.Append(layer, []float32{float32(token)}, []float32{float32(token + 100)}); err != nil {
		t.Fatalf("Append layer %d token %d: %v", layer, token, err)
	}
}

func snapshotLayeredF32KVState(t *testing.T, cache *LayeredF32KV) []layeredF32KVState {
	t.Helper()
	state := make([]layeredF32KVState, len(cache.stores))
	for layer := range cache.stores {
		k, v, startToken, err := cache.MaterializeLayer(layer)
		if err != nil {
			t.Fatalf("MaterializeLayer(%d): %v", layer, err)
		}
		state[layer] = layeredF32KVState{
			startToken: startToken,
			k:          append([]float32(nil), k...),
			v:          append([]float32(nil), v...),
		}
	}
	return state
}

func requireLayeredSnapshot(t *testing.T, cache *LayeredF32KV, want []layeredF32KVState) {
	t.Helper()
	got := snapshotLayeredF32KVState(t, cache)
	if len(got) != len(want) {
		t.Fatalf("snapshot layers=%d want %d", len(got), len(want))
	}
	for layer := range want {
		if got[layer].startToken != want[layer].startToken {
			t.Fatalf("layer %d startToken=%d want %d", layer, got[layer].startToken, want[layer].startToken)
		}
		if !sameFloat32s(got[layer].k, want[layer].k) || !sameFloat32s(got[layer].v, want[layer].v) {
			t.Fatalf("layer %d materialized mismatch gotK=%v wantK=%v gotV=%v wantV=%v", layer, got[layer].k, want[layer].k, got[layer].v, want[layer].v)
		}
	}
}

func requireLayeredMaterialized(t *testing.T, cache *LayeredF32KV, layer, wantStart int, wantK, wantV []float32) {
	t.Helper()
	k, v, startToken, err := cache.MaterializeLayer(layer)
	if err != nil {
		t.Fatalf("MaterializeLayer(%d): %v", layer, err)
	}
	if startToken != wantStart {
		t.Fatalf("MaterializeLayer(%d).StartToken=%d want %d", layer, startToken, wantStart)
	}
	if !sameFloat32s(k, wantK) {
		t.Fatalf("MaterializeLayer(%d).K=%v want %v", layer, k, wantK)
	}
	if !sameFloat32s(v, wantV) {
		t.Fatalf("MaterializeLayer(%d).V=%v want %v", layer, v, wantV)
	}
}

func TestEstimateLayeredF32KVBytes4K16K32K(t *testing.T) {
	configs := []LayerF32KVConfig{{Dim: 128}, {Dim: 128, Sliding: true, SlidingWindow: 4096}}
	maxChunk := 512
	ringWant := int64((4096 + 512) * 128 * 8)
	for _, maxContext := range []int{4096, 16384, 32768} {
		linearBytes, ringBytes, err := EstimateLayeredF32KVBytes(configs, maxContext, maxChunk)
		if err != nil {
			t.Fatalf("EstimateLayeredF32KVBytes(%d): %v", maxContext, err)
		}
		linearWant := int64(maxContext * 128 * 8)
		if linearBytes != linearWant || ringBytes != ringWant {
			t.Fatalf("EstimateLayeredF32KVBytes(%d) linear=%d want %d ring=%d want %d", maxContext, linearBytes, linearWant, ringBytes, ringWant)
		}
	}
}
