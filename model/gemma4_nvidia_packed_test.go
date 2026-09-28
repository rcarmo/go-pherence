package model

import (
	"context"
	"math"
	"reflect"
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestGemma4PackedPrefillMatchesIndependent(t *testing.T) {
	if !nvidia.SgemmReady() {
		if nvidia.Available() {
			t.Fatal("device available but PTX not ready")
		}
		t.Skip("CUDA unavailable")
	}
	for _, window := range []int{0, 3} {
		t.Run(string(rune('0'+window)), func(t *testing.T) {
			m := newGemma4NVIDIAQuantTestModel(t)
			if window > 0 {
				m.Config.SlidingWindow = window
				m.Config.LayerTypes = []string{"sliding_attention"}
			}
			g, err := NewGemma4NVIDIA(m)
			if err != nil {
				t.Fatal(err)
			}
			defer g.Close()
			prefix, err := g.PrefillPreparedCapacity(context.Background(), []int{1, 2}, 12, 1, 1)
			if err != nil {
				t.Fatal(err)
			}
			defer prefix.Close()
			paths := [][]int{{0, 1, 2, 0}, {2, 2, 1, 0, 1}, {1, 0, 1, 2, 1, 2}}
			candidates := [][]int{{0, 1, 2}, {2, 0}, {1, 2}}
			want := make([][]float32, len(paths))
			for i := range paths {
				want[i], err = g.ScorePrefixedSingleBranch(context.Background(), prefix, paths[i], candidates[i])
				if err != nil {
					t.Fatal(err)
				}
			}
			before := make([]float32, prefix.arena.trunkLen*prefix.arena.kvDim[0])
			if err := prefix.arena.trunkK[0].Download(before); err != nil {
				t.Fatal(err)
			}
			got, err := g.ScorePrefixedPacked(context.Background(), prefix, paths, candidates)
			if err != nil {
				t.Fatal(err)
			}
			for i := range got {
				assertFloat32RowsRelative(t, got[i], want[i], 1e-2, 5e-3)
			}
			after := make([]float32, len(before))
			if err := prefix.arena.trunkK[0].Download(after); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(before, after) {
				t.Fatal("packed scoring mutated prefix arena")
			}
			paths[0] = []int{2, 2, 2, 2}
			changed, err := g.ScorePrefixedPacked(context.Background(), prefix, paths, candidates)
			if err != nil {
				t.Fatal(err)
			}
			for i := 1; i < len(paths); i++ {
				assertFloat32RowsRelative(t, changed[i], got[i], 1e-6, 1e-6)
			}
			reverse, err := g.ScorePrefixedPacked(context.Background(), prefix, [][]int{paths[2], paths[1], paths[0]}, [][]int{candidates[2], candidates[1], candidates[0]})
			if err != nil {
				t.Fatal(err)
			}
			for i := range reverse {
				assertFloat32RowsRelative(t, reverse[i], changed[len(paths)-1-i], 1e-6, 1e-6)
			}
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if result, err := g.ScorePrefixedPacked(ctx, prefix, paths, candidates); err == nil || result != nil {
				t.Fatal("cancelled call accepted")
			}
			for _, bad := range [][][]int{nil, {{-1}}, {make([]int, Gemma4PackedRows+1)}} {
				if result, err := g.ScorePrefixedPacked(context.Background(), prefix, bad, [][]int{{0}}); err == nil || result != nil {
					t.Fatal("invalid paths accepted")
				}
			}
			if _, err := g.ScorePrefixedPacked(context.Background(), prefix, paths, candidates); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGemma4SelectedLogitTransforms(t *testing.T) {
	g := &Gemma4NVIDIA{model: &LlamaModel{Config: LlamaConfig{FinalLogitSoftcapping: 3}, SuppressTokens: []int{7}}}
	got := []float32{2, 9, -4}
	g.transformSelectedLogits(got, []int{1, 7, 3})
	want := []float32{2, 9, -4}
	applyLlamaFinalLogitSoftcap(want, 3)
	if got[0] != want[0] || !math.IsInf(float64(got[1]), -1) || got[2] != want[2] {
		t.Fatalf("transforms: %v", got)
	}
}

func TestGemma4DeviceWorkReusesAndFrees(t *testing.T) {
	if !nvidia.SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	var w gemma4DeviceWork
	w.begin()
	a, b := w.alloc(16), w.alloc(32)
	if w.err != nil {
		t.Fatal(w.err)
	}
	w.begin()
	if w.alloc(8) != a || w.alloc(32) != b {
		t.Fatal("scratch not reused")
	}
	w.begin()
	c := w.alloc(64)
	if w.err != nil {
		t.Fatal(w.err)
	}
	if c.Size < 64*4 || a.Ptr != 0 {
		t.Fatal("growth did not replace old buffer")
	}
	w.free()
	if c.Ptr != 0 || b.Ptr != 0 || len(w.buffers) != 0 {
		t.Fatal("scratch leak")
	}
	w.free()
}

func TestGemma4PackedTreeGroupsIsolation(t *testing.T) {
	if !nvidia.SgemmReady() {
		t.Skip("CUDA unavailable")
	}
	for _, window := range []int{0, 3} {
		m := newGemma4NVIDIAQuantTestModel(t)
		if window > 0 {
			m.Config.SlidingWindow = window
			m.Config.LayerTypes = []string{"sliding_attention"}
		}
		g, err := NewGemma4NVIDIA(m)
		if err != nil {
			t.Fatal(err)
		}
		prefix, err := g.PrefillPreparedCapacity(context.Background(), []int{1, 2}, 18, 1, 1)
		if err != nil {
			t.Fatal(err)
		}
		contexts := [][]int{{0, 1, 2, 1}, {2, 1, 0, 2, 0}}
		branches := [][]int{{0}, {1, 0}, {2, 0, 1}}
		ids := [][]int{{0, 1, 2}, {1, 2}, {2, 0}}
		want := make([][][]float32, len(contexts))
		for i, c := range contexts {
			want[i] = make([][]float32, len(branches))
			for j, b := range branches {
				path := append(append([]int{}, c...), b...)
				want[i][j], err = g.ScorePrefixedSingleBranch(context.Background(), prefix, path, ids[j])
				if err != nil {
					t.Fatal(err)
				}
			}
		}
		got, err := g.ScorePrefixedTreeGroups(context.Background(), prefix, contexts, branches, ids)
		if err != nil {
			t.Fatal(err)
		}
		for i := range got {
			for j := range got[i] {
				assertFloat32RowsRelative(t, got[i][j], want[i][j], 1e-2, 5e-3)
			}
		}
		branches[0] = []int{2}
		changed, err := g.ScorePrefixedTreeGroups(context.Background(), prefix, contexts, branches, ids)
		if err != nil {
			t.Fatal(err)
		}
		for i := range got {
			for j := 1; j < len(branches); j++ {
				assertFloat32RowsRelative(t, changed[i][j], got[i][j], 1e-6, 1e-6)
			}
		}
		contexts[0] = []int{2, 2, 2, 2}
		again, err := g.ScorePrefixedTreeGroups(context.Background(), prefix, contexts, branches, ids)
		if err != nil {
			t.Fatal(err)
		}
		for j := range branches {
			assertFloat32RowsRelative(t, again[1][j], changed[1][j], 1e-6, 1e-6)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if out, err := g.ScorePrefixedTreeGroups(ctx, prefix, contexts, branches, ids); err == nil || out != nil {
			t.Fatal("cancelled tree accepted")
		}
		if _, err := g.ScorePrefixedTreeGroups(context.Background(), prefix, contexts, branches, ids); err != nil {
			t.Fatal(err)
		}
		prefix.Close()
		g.Close()
	}
}
