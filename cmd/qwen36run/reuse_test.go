package main

import (
	"testing"

	"github.com/rcarmo/go-pherence/model/qwen"
	"github.com/rcarmo/go-pherence/runtime/kv"
)

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
