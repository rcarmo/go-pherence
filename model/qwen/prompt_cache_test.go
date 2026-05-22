package qwen

import "testing"

func TestPromptCacheStoreFindAndStats(t *testing.T) {
	c := NewPromptCache(1 << 20)
	snap := PromptSnapshot{EndPos: 2, Next: 42, State: Qwen35BaseForwardState{Pos: 2, FullK: [][]float32{{1}}, FullV: [][]float32{{2}}}, Hidden: []float32{3}, PreNorm: []float32{4}}
	if !c.Store("m", "l", "mlx4", []int{1, 2}, 2, snap) {
		t.Fatal("store failed")
	}
	got, ok := c.FindLongest("m", "l", "mlx4", []int{1, 2, 3}, 2)
	if !ok || got.EndPos != 2 || got.Next != 42 || got.State.FullK[0][0] != 1 {
		t.Fatalf("bad restore got=%+v ok=%v", got, ok)
	}
	got.State.FullK[0][0] = 99
	got2, ok := c.FindLongest("m", "l", "mlx4", []int{1, 2}, 2)
	if !ok || got2.State.FullK[0][0] != 1 {
		t.Fatalf("snapshot not cloned: %+v", got2)
	}
	st := c.Stats()
	if st.Entries != 1 || st.SidecarEntries != 1 || st.UsedBytes <= 0 || st.MaxBytes <= 0 {
		t.Fatalf("bad stats: %+v", st)
	}
}

func TestPromptCacheOversizedStoreEvictsSidecar(t *testing.T) {
	c := NewPromptCache(4)
	if c.Store("m", "l", "mlx4", []int{1}, 1, PromptSnapshot{EndPos: 1, State: Qwen35BaseForwardState{Pos: 1}, Hidden: []float32{1, 2}}) {
		t.Fatal("oversized store reported retained")
	}
	st := c.Stats()
	if st.Entries != 0 || st.SidecarEntries != 0 || st.UsedBytes != 0 {
		t.Fatalf("oversized store leaked: %+v", st)
	}
}
