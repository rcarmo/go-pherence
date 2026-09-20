package qwen

import (
	"github.com/rcarmo/go-pherence/runtime/kv"
	"sync"
	"testing"
)

func TestPromptCacheAdmitsSidecarOnceAndDropsSpareCapacity(t *testing.T) {
	row := make([]float32, 2, 4096)
	row[0] = 7
	snap := PromptSnapshot{State: Qwen35BaseForwardState{Pos: 2, FullK: [][]float32{row}, FullV: [][]float32{row, row}}, EndPos: 2}
	size, err := promptSnapshotBytes(snap)
	if err != nil {
		t.Fatal(err)
	}
	key := PromptPrefixKey("m", "l", "f32", []int{1, 2}, 2)
	budget, err := kv.EstimateChunkEntryBytes(key, 2, kv.Snapshot{}, size)
	if err != nil {
		t.Fatal(err)
	}
	c := NewPromptCache(budget)
	if !c.Store("m", "l", "f32", []int{1, 2}, 2, snap) {
		t.Fatal("exact admitted budget rejected")
	}
	if st := c.Stats(); st.UsedBytes != budget || st.Entries != 1 || st.SidecarEntries != 1 {
		t.Fatal(st)
	}
	got, ok := c.FindLongest("m", "l", "f32", []int{1, 2}, 2)
	if !ok || len(got.State.FullV) != 2 || cap(got.State.FullK[0]) > 8 {
		t.Fatal("lost V or retained spare generation capacity")
	}
	row[0] = 9
	got.State.FullK[0][0] = 10
	again, _ := c.FindLongest("m", "l", "f32", []int{1, 2}, 2)
	if again.State.FullK[0][0] != 7 {
		t.Fatal("alias")
	}
	if NewPromptCache(0).Store("m", "l", "f32", []int{1, 2}, 2, snap) {
		t.Fatal("zero budget enabled")
	}
	if NewPromptCache(budget-1).Store("m", "l", "f32", []int{1, 2}, 2, snap) {
		t.Fatal("metadata omitted")
	}
}
func TestPromptCacheRejectPreservesValueAndConcurrentCalls(t *testing.T) {
	c := NewPromptCache(4096)
	tokens := []int{1}
	snap := PromptSnapshot{Hidden: []float32{7}}
	if !c.Store("m", "l", "f32", tokens, 1, snap) {
		t.Fatal("store")
	}
	if c.Store("m", "l", "f32", tokens, 1, PromptSnapshot{Hidden: make([]float32, 4096)}) {
		t.Fatal("over budget")
	}
	got, ok := c.FindLongest("m", "l", "f32", tokens, 1)
	if !ok || got.Hidden[0] != 7 {
		t.Fatal("lost entry")
	}
	var wg sync.WaitGroup
	for i := 0; i < 6; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 30; j++ {
				c.Store("m", "l", "f32", tokens, 1, snap)
				c.FindLongest("m", "l", "f32", tokens, 1)
				c.Stats()
				c.PruneSidecar()
			}
		}()
	}
	wg.Wait()
}
