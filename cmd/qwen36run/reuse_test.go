package main

import (
	"testing"

	"github.com/rcarmo/go-pherence/model/qwen"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestQwenStorePromptPrefixStoresForwardState(t *testing.T) {
	qwenPromptStateCache = qwen.NewPromptCache(1 << 20)
	modelID, layout := "m", "l"
	state := qwen.Qwen35BaseForwardState{Pos: 2, FullK: [][]float32{{1, 2}}, FullV: [][]float32{{3, 4}}, Linear: []qwen.Qwen35LinearAttentionState{{Conv: []float32{5}, SSM: []float32{6}, Pos: 2}}}
	if !qwenStorePromptPrefix(modelID, layout, []int{1, 2}, 2, qwen.PromptSnapshot{EndPos: 2, Next: 20, State: state, Hidden: []float32{7}, PreNorm: []float32{8}}) {
		t.Fatal("store failed")
	}
	got, ok := qwenFindLongestPromptPrefix(modelID, layout, []int{1, 2, 3}, 2)
	if !ok || got.EndPos != 2 || got.State.Pos != 2 || len(got.State.FullK) != 1 || got.State.FullK[0][0] != 1 || got.State.Linear[0].Conv[0] != 5 {
		t.Fatalf("stored state not restored: %+v ok=%v", got, ok)
	}
	got.State.FullK[0][0] = 99
	got2, ok := qwenFindLongestPromptPrefix(modelID, layout, []int{1, 2}, 2)
	if !ok || got2.State.FullK[0][0] != 1 {
		t.Fatalf("stored state was not isolated: %+v", got2)
	}
}

func TestQwenVerifyFloatSamples(t *testing.T) {
	if !qwenVerifyFloatSamples([]float32{1, 2, 3}, []float32{1, 2.001, 3}, 0.01) {
		t.Fatal("expected tolerant match")
	}
	if qwenVerifyFloatSamples([]float32{1, 2, 3}, []float32{1, 2.1, 3}, 0.01) {
		t.Fatal("expected mismatch")
	}
	if qwenVerifyFloatSamples([]float32{1}, []float32{1, 2}, 0) {
		t.Fatal("expected length mismatch")
	}
}

func TestShouldStoreQwenPromptPrefixPolicy(t *testing.T) {
	if !shouldStoreQwenPromptPrefix(2, 5, 2, 1, false, 1) {
		t.Fatal("expected chunk store")
	}
	if shouldStoreQwenPromptPrefix(2, 5, 2, 2, false, 1) {
		t.Fatal("store-every should skip chunk 1")
	}
	if !shouldStoreQwenPromptPrefix(4, 5, 2, 2, false, 1) {
		t.Fatal("store-every should store chunk 2")
	}
	if shouldStoreQwenPromptPrefix(4, 5, 2, 1, true, 1) {
		t.Fatal("final-only stored non-final prefix")
	}
	if !shouldStoreQwenPromptPrefix(5, 5, 2, 100, true, 1) {
		t.Fatal("final-only skipped final prefix")
	}
	if shouldStoreQwenPromptPrefix(2, 5, 2, 1, false, 3) {
		t.Fatal("min tokens not enforced")
	}
}

func TestKVSkippedPrefillAccounting(t *testing.T) {
	inputLen, repeat, prefill := 3, 2, 3
	total := inputLen * repeat
	skipped := total - prefill
	if skipped != 3 {
		t.Fatalf("skipped=%d want 3", skipped)
	}
	eff := float64(skipped) / float64(total)
	if eff != 0.5 {
		t.Fatalf("efficiency=%v want 0.5", eff)
	}
}

func TestQwenPromptStateCacheBudgetAccessors(t *testing.T) {
	c := kv.NewChunkCache(1234)
	if c.MaxBytes() != 1234 || c.UsedBytes() != 0 || c.Len() != 0 {
		t.Fatalf("unexpected cache accessors max=%d used=%d len=%d", c.MaxBytes(), c.UsedBytes(), c.Len())
	}
}

func TestNativeMTPDraftStepsForRemaining(t *testing.T) {
	cases := []struct {
		mtpSteps  int
		remaining int
		want      int
	}{
		{2, 0, 0},
		{2, 1, 0},
		{2, 2, 1},
		{2, 3, 2},
		{4, 3, 2},
		{0, 3, 0},
	}
	for _, tc := range cases {
		if got := nativeMTPDraftStepsForRemaining(tc.mtpSteps, tc.remaining); got != tc.want {
			t.Fatalf("steps(%d,%d)=%d want %d", tc.mtpSteps, tc.remaining, got, tc.want)
		}
	}
}

func TestShouldFallbackNativeMTP(t *testing.T) {
	stats := mtpGenerateStats{Accepted: 2, Drafted: 4, Rounds: 4}
	if shouldFallbackNativeMTP(stats, false, 0.75, 4) {
		t.Fatal("disabled adaptive fallback triggered")
	}
	if shouldFallbackNativeMTP(stats, true, 0.75, 5) {
		t.Fatal("fallback triggered before warmup")
	}
	if !shouldFallbackNativeMTP(stats, true, 0.75, 4) {
		t.Fatal("fallback did not trigger below threshold")
	}
	if shouldFallbackNativeMTP(mtpGenerateStats{Accepted: 3, Drafted: 4, Rounds: 4}, true, 0.75, 4) {
		t.Fatal("fallback triggered at threshold")
	}
	if shouldFallbackNativeMTP(mtpGenerateStats{Accepted: 0, Drafted: 0, Rounds: 4}, true, 0.75, 4) {
		t.Fatal("fallback triggered with no drafts")
	}
}

func TestQwenStorePromptPrefixReportsEvictedStore(t *testing.T) {
	qwenPromptStateCache = qwen.NewPromptCache(4)
	if qwenStorePromptPrefix("m", "l", []int{1}, 1, qwen.PromptSnapshot{EndPos: 1, State: qwen.Qwen35BaseForwardState{Pos: 1}, Hidden: []float32{1, 2}}) {
		t.Fatal("oversized store reported success")
	}
	if qwenPromptStateCache.Stats().SidecarEntries != 0 || qwenPromptStateCache.Stats().Entries != 0 {
		t.Fatalf("oversized store left entries sidecar=%d cache=%d", qwenPromptStateCache.Stats().SidecarEntries, qwenPromptStateCache.Stats().Entries)
	}
}

func TestQwenPromptSnapshotForBudgetIncludesForwardState(t *testing.T) {
	snap := qwen.PromptSnapshot{
		State:   qwen.Qwen35BaseForwardState{Pos: 3, FullK: [][]float32{{1, 2}}, FullV: [][]float32{{3, 4}}, Linear: []qwen.Qwen35LinearAttentionState{{Conv: []float32{5, 6, 7}, SSM: []float32{8, 9}, Pos: 3}}},
		Hidden:  []float32{10},
		PreNorm: []float32{11, 12},
	}
	budget := qwen.PromptSnapshotForBudget(snap)
	if len(budget.Layers) != 2 || len(budget.Layers[0].K) != 2 || len(budget.Layers[0].V) != 2 || len(budget.Layers[1].K) != 3 || len(budget.Layers[1].V) != 2 {
		t.Fatalf("budget snapshot did not include forward state: %+v", budget)
	}
	bytes, err := kv.EstimateSnapshotBytes(budget)
	if err != nil {
		t.Fatal(err)
	}
	wantFloats := 2 + 2 + 3 + 2 + 3
	if bytes != int64(wantFloats*4) {
		t.Fatalf("bytes=%d want %d", bytes, wantFloats*4)
	}
}

func TestQwenStorePromptPrefixPrunesEvictedSidecar(t *testing.T) {
	qwenPromptStateCache = qwen.NewPromptCache(4)
	modelID, layout := "m", "l"
	_ = qwenStorePromptPrefix(modelID, layout, []int{1}, 1, qwen.PromptSnapshot{EndPos: 1, State: qwen.Qwen35BaseForwardState{Pos: 1}, Hidden: []float32{1, 2}})
	_ = qwenStorePromptPrefix(modelID, layout, []int{1, 2}, 1, qwen.PromptSnapshot{EndPos: 2, State: qwen.Qwen35BaseForwardState{Pos: 2}, Hidden: []float32{3, 4}})
	if qwenPromptStateCache.Stats().SidecarEntries != qwenPromptStateCache.Stats().Entries {
		t.Fatalf("sidecar/cache mismatch sidecar=%d cache=%d", qwenPromptStateCache.Stats().SidecarEntries, qwenPromptStateCache.Stats().Entries)
	}
}

func TestQwenFindLongestPromptPrefixFromPrimedPrompt(t *testing.T) {
	qwenPromptStateCache = qwen.NewPromptCache(1 << 20)
	modelID, layout := "m", "l"
	prime := []int{10, 20}
	extended := []int{10, 20, 30}
	if !qwenStorePromptPrefix(modelID, layout, prime, 2, qwen.PromptSnapshot{EndPos: 2, Next: 200, State: qwen.Qwen35BaseForwardState{Pos: 2}, Hidden: []float32{1}, PreNorm: []float32{2}}) {
		t.Fatal("store primed prefix")
	}
	got, ok := qwenFindLongestPromptPrefix(modelID, layout, extended, 2)
	if !ok || got.EndPos != 2 || got.Next != 200 || got.State.Pos != 2 {
		t.Fatalf("primed prefix not restored: %+v ok=%v", got, ok)
	}
	if _, ok := qwenFindLongestPromptPrefix(modelID, layout, []int{10, 99, 30}, 2); ok {
		t.Fatal("unexpected unrelated prompt hit")
	}
}

func TestQwenFindLongestPromptPrefix(t *testing.T) {
	qwenPromptStateCache = qwen.NewPromptCache(1 << 20)
	modelID, layout := "m", "l"
	if !qwenStorePromptPrefix(modelID, layout, []int{1, 2}, 2, qwen.PromptSnapshot{EndPos: 2, Next: 20, State: qwen.Qwen35BaseForwardState{Pos: 2}}) {
		t.Fatal("store short prefix")
	}
	if !qwenStorePromptPrefix(modelID, layout, []int{1, 2, 3, 4}, 2, qwen.PromptSnapshot{EndPos: 4, Next: 40, State: qwen.Qwen35BaseForwardState{Pos: 4}}) {
		t.Fatal("store long prefix")
	}
	got, ok := qwenFindLongestPromptPrefix(modelID, layout, []int{1, 2, 3, 4, 5}, 2)
	if !ok || got.EndPos != 4 || got.Next != 40 {
		t.Fatalf("longest=%+v ok=%v", got, ok)
	}
	got, ok = qwenFindLongestPromptPrefix(modelID, layout, []int{1, 2, 9}, 2)
	if !ok || got.EndPos != 2 || got.Next != 20 {
		t.Fatalf("partial=%+v ok=%v", got, ok)
	}
	if _, ok := qwenFindLongestPromptPrefix(modelID, layout, []int{9, 9}, 2); ok {
		t.Fatal("unexpected prefix hit")
	}
}
