package main

import (
	"testing"

	"github.com/rcarmo/go-pherence/model/qwen"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

func TestQwenStorePromptPrefixStoresForwardState(t *testing.T) {
	qwenPromptStateCache = kv.NewChunkCache(1 << 20)
	qwenPromptStateSidecar = map[kv.ChunkKey]qwenPromptStateSnapshot{}
	modelID, layout := "m", "l"
	state := qwen.Qwen35BaseForwardState{Pos: 2, FullK: [][]float32{{1, 2}}, FullV: [][]float32{{3, 4}}, Linear: []qwen.Qwen35LinearAttentionState{{Conv: []float32{5}, SSM: []float32{6}, Pos: 2}}}
	qwenStorePromptPrefix(modelID, layout, []int{1, 2}, 2, qwenPromptStateSnapshot{EndPos: 2, Next: 20, State: state, Hidden: []float32{7}, PreNorm: []float32{8}})
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

func TestQwenStorePromptPrefixPrunesEvictedSidecar(t *testing.T) {
	qwenPromptStateCache = kv.NewChunkCache(4)
	qwenPromptStateSidecar = map[kv.ChunkKey]qwenPromptStateSnapshot{}
	modelID, layout := "m", "l"
	qwenStorePromptPrefix(modelID, layout, []int{1}, 1, qwenPromptStateSnapshot{EndPos: 1, State: qwen.Qwen35BaseForwardState{Pos: 1}, Hidden: []float32{1, 2}})
	qwenStorePromptPrefix(modelID, layout, []int{1, 2}, 1, qwenPromptStateSnapshot{EndPos: 2, State: qwen.Qwen35BaseForwardState{Pos: 2}, Hidden: []float32{3, 4}})
	if len(qwenPromptStateSidecar) != qwenPromptStateCache.Len() {
		t.Fatalf("sidecar/cache mismatch sidecar=%d cache=%d", len(qwenPromptStateSidecar), qwenPromptStateCache.Len())
	}
}

func TestQwenFindLongestPromptPrefix(t *testing.T) {
	qwenPromptStateCache = kv.NewChunkCache(1 << 20)
	qwenPromptStateSidecar = map[kv.ChunkKey]qwenPromptStateSnapshot{}
	modelID, layout := "m", "l"
	qwenStorePromptPrefix(modelID, layout, []int{1, 2}, 2, qwenPromptStateSnapshot{EndPos: 2, Next: 20, State: qwen.Qwen35BaseForwardState{Pos: 2}})
	qwenStorePromptPrefix(modelID, layout, []int{1, 2, 3, 4}, 2, qwenPromptStateSnapshot{EndPos: 4, Next: 40, State: qwen.Qwen35BaseForwardState{Pos: 4}})
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
