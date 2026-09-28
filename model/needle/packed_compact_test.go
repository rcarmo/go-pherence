package needle

import (
	"context"
	"math"
	"runtime"
	"strings"
	"sync"
	"testing"
)

// The compact model is derived only after loading the fixture in the existing
// decoded format; this checks steady-state model data, not load-time peak RSS.
func TestCompactPackedConcurrentOwnership(t *testing.T) {
	m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	compact, err := m.CompactPacked()
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{2, 7, 4, 9}
	opts := Options{Packed: true}
	want, err := m.Forward(ids, opts)
	if err != nil {
		t.Fatal(err)
	}
	// The derived view shares only immutable CQ and dense-required storage;
	// dropping the parent must not invalidate either reference.
	m = nil
	runtime.GC()
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for call := 0; call < 10; call++ {
				got, e := compact.Forward(ids, opts)
				if e != nil || len(got) != len(want) {
					t.Errorf("forward error=%v size=%d", e, len(got))
					return
				}
				for i, v := range got {
					if math.Float32bits(v) != math.Float32bits(want[i]) {
						t.Errorf("forward[%d] differs", i)
						return
					}
				}
				clear(got)
				d, e := compact.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: opts})
				if e != nil {
					t.Error(e)
					return
				}
				for _, id := range ids {
					if _, e = d.Step(context.Background(), id); e != nil {
						t.Error(e)
						return
					}
				}
			}
		}()
	}
	wg.Wait()
	got, err := compact.Forward(ids, opts)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if math.Float32bits(v) != math.Float32bits(want[i]) {
			t.Fatal("caller changed retained weights")
		}
	}
}

func TestCompactPackedTinyArchiveParityAndGuards(t *testing.T) {
	m, _, err := LoadArchive("../../loader/needle/testdata/needle3.cact")
	if err != nil {
		t.Fatal(err)
	}
	before := m.DecodedBytes()
	compact, err := m.CompactPacked()
	if err != nil {
		t.Fatal(err)
	}
	if before != 25012 || compact.DecodedBytes() != 10420 || m.DecodedBytes() != before || compact.PackedBytes() != m.PackedBytes() {
		t.Fatalf("decoded bytes original=%d compact=%d packed original=%d compact=%d", m.DecodedBytes(), compact.DecodedBytes(), m.PackedBytes(), compact.PackedBytes())
	}
	if len(compact.packedShapes) == 0 || compact.Checkpoint() != nil || m.Checkpoint() == nil {
		t.Fatal("compact export or original mutation")
	}
	if _, err = compact.CompactPacked(); err == nil {
		t.Fatal("accepted double compaction")
	}
	if _, err = planPackedOnly(compact); err == nil {
		t.Fatal("planner accepted missing decoded tensors in compact view")
	}
	if _, err = compact.Forward([]int{2}, Options{Packed: true, MaxWorkBytes: 1}); err == nil {
		t.Fatal("ignored compact execution workspace limit")
	}
	if _, err = compact.SliceDepth(2); err == nil {
		t.Fatal("accepted depth slice")
	}
	if _, err = compact.Forward([]int{2}, Options{}); err == nil || !strings.Contains(err.Error(), "requires packed execution") {
		t.Fatalf("accepted dense forward: %v", err)
	}
	if _, err = compact.NewDecoder(DecoderOptions{Capacity: 12}); err == nil {
		t.Fatal("accepted dense cached decoder")
	}
	if _, err = compact.Head([]int{2}, Embedding, Options{}); err == nil {
		t.Fatal("accepted dense head")
	}
	ids := []int{2, 7, 4, 9, 3, 6, 5, 8, 7, 3, 5, 2}
	opts := Options{Packed: true}
	for i := 1; i <= len(ids); i++ {
		want, err := m.Forward(ids[:i], opts)
		if err != nil {
			t.Fatal(err)
		}
		got, err := compact.Forward(ids[:i], opts)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "compact forward", got, want, 1e-4, 5e-3)
	}
	decoder, err := compact.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: opts})
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := m.NewDecoder(DecoderOptions{Capacity: len(ids), Execution: opts})
	if err != nil {
		t.Fatal(err)
	}
	for _, token := range ids {
		want, err := baseline.Step(context.Background(), token)
		if err != nil {
			t.Fatal(err)
		}
		got, err := decoder.Step(context.Background(), token)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "compact cached", got, want, 1e-4, 5e-3)
	}
	for _, kind := range []HeadKind{Embedding, Confidence, Router} {
		want, err := m.Head(ids, kind, opts)
		if err != nil {
			t.Fatal(err)
		}
		got, err := compact.Head(ids, kind, opts)
		if err != nil {
			t.Fatal(err)
		}
		compare(t, "compact head", got, want, 2e-5, 5e-3)
	}
}
