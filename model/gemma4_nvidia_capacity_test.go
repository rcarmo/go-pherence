package model

import (
	"context"
	"errors"
	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
	"reflect"
	"testing"
)

func TestGemma4ArenaSizeValidation(t *testing.T) {
	n, e := gemma4ArenaBytes([]int{32, 0, 64}, 4096, 12, 8)
	if e != nil || n != (4096+96)*96*8 {
		t.Fatalf("%d %v", n, e)
	}
	for _, x := range [][3]int{{0, 1, 1}, {32769, 1, 1}, {1, 257, 1}, {1, 1, 32769}} {
		if _, e := gemma4ArenaBytes([]int{32}, x[0], x[1], x[2]); !errors.Is(e, ErrGemma4ContextCapacity) {
			t.Fatal(e)
		}
	}
	if _, e := gemma4ArenaBytes([]int{int(^uint(0) >> 1)}, 32768, 256, 32768); !errors.Is(e, ErrGemma4ContextCapacity) {
		t.Fatal("overflow accepted")
	}
}
func TestSlidingKVTailAndClone(t *testing.T) {
	if !nvidia.SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	const total, window, dim = 4096, 128, 4
	a, e := allocGemma4NVIDIAKVArenaWindowed([]int{dim}, total, 2, 8, []int{window})
	if e != nil {
		t.Fatal(e)
	}
	defer a.free()
	b, e := nvidia.Malloc(Gemma4PackedRows * dim)
	if e != nil {
		t.Fatal(e)
	}
	defer b.Free()
	for pos := 0; pos < total; pos += 512 {
		x := make([]float32, 512*dim)
		for i := range x {
			x[i] = float32(pos*dim + i)
		}
		if e = b.Upload(x); e != nil {
			t.Fatal(e)
		}
		if e = a.appendTrunkRows(0, pos, 512, b, b); e != nil {
			t.Fatal(e)
		}
	}
	base := a.trunkBase[0]
	if base == 0 || total-base > window+512 {
		t.Fatalf("base %d", base)
	}
	want := make([]float32, (total-base)*dim)
	for i := range want {
		want[i] = float32(base*dim + i)
	}
	got := make([]float32, len(want))
	if e = a.trunkK[0].Download(got); e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("compacted KV changed")
	}
	a.trunkUsed = total
	clone, e := a.cloneTrunk(total+2, 2, 8)
	if e != nil {
		t.Fatal(e)
	}
	defer clone.free()
	if clone.trunkBase[0] != base {
		t.Fatal("clone lost offset")
	}
	if e = clone.trunkK[0].Download(got); e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("clone changed tail")
	}
	c := &Gemma4NVIDIAContext{arena: a}
	if c.CanReusePrefix() {
		t.Fatal("evicted prefix reused")
	}
	if e = a.appendTrunkRows(0, 0, 1, b, b); e == nil {
		t.Fatal("rewind silently accepted")
	}
}

// Compare compacted storage with full KV storage under the same sliding mask.
// This exercises absolute RoPE, prefill chunk boundaries, packed prefix offsets
// and branch-local suffixes without allocating a released checkpoint.
func TestGemma4SlidingPrefillMatchesFullKV(t *testing.T) {
	if !nvidia.SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	m := newGemma4NVIDIAQuantTestModel(t)
	m.Config.MaxSeqLen = 8192
	m.Config.SlidingWindow = 128
	m.Config.LayerTypes = []string{"sliding_attention"}
	g, err := NewGemma4NVIDIA(m)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	const prefixLen = 2407
	tokens := make([]int, prefixLen)
	for i := range tokens {
		tokens[i] = i % 3
	}
	compact, err := g.PrefillPreparedCapacity(context.Background(), tokens, prefixLen+16, 3, 8)
	if err != nil {
		t.Fatal(err)
	}
	defer compact.Close()
	fullArena, err := allocGemma4NVIDIAKVArenaShape(compact.arena.kvDim, prefixLen+16, 3, 8)
	if err != nil {
		t.Fatal(err)
	}
	full := &Gemma4NVIDIAContext{owner: g, tokens: tokens, arena: fullArena}
	defer full.Close()
	for pos := 0; pos < len(tokens); pos += 512 {
		end := min(len(tokens), pos+512)
		if err = g.runPrefillBatch(context.Background(), tokens[pos:end], pos, fullArena, nil); err != nil {
			t.Fatal(err)
		}
	}
	fullArena.trunkUsed = prefixLen
	paths := [][]int{{0, 1}, {2}, {1, 0, 2}}
	ids := [][]int{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}
	want, err := g.ScorePrefixedPacked(context.Background(), full, paths, ids)
	if err != nil {
		t.Fatal(err)
	}
	got, err := g.ScorePrefixedPacked(context.Background(), compact, paths, ids)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		assertFloat32RowsRelative(t, got[i], want[i], 1e-6, 1e-6)
	}
	wantBranches, err := full.ScoreIndependentBranches(context.Background(), paths)
	if err != nil {
		t.Fatal(err)
	}
	gotBranches, err := compact.ScoreIndependentBranches(context.Background(), paths)
	if err != nil {
		t.Fatal(err)
	}
	for i := range wantBranches.Logits {
		assertFloat32RowsRelative(t, gotBranches.Logits[i], wantBranches.Logits[i], 1e-6, 1e-6)
	}
	if compact.CanReusePrefix() {
		t.Fatal("compacted cache reused")
	}
}
