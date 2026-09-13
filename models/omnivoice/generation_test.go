package omnivoice

import (
	"context"
	"encoding/json"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"os"
	"reflect"
	"testing"
)

func TestGenerationDeterminismAndAllocation(t *testing.T) {
	b, f := loadBackboneFixture(t)
	uncond, err := NewBackbone(b.weights, 2)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultGenerationConfig()
	cfg.Steps = 4
	g, err := NewGeneration(b, uncond, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids := append([]int(nil), f.IDs...)
	uid := make([]int, 4)
	ua := []bool{true, true}
	out := make([]int, 4)
	if err = g.GenerateInto(context.Background(), out, ids, f.AudioMask, uid, ua); err != nil {
		t.Fatal(err)
	}
	expected := append([]int(nil), out...)
	for _, v := range out {
		if v < 0 || v >= 4 {
			t.Fatalf("unfilled/invalid token %d", v)
		}
	}
	allocations := testing.AllocsPerRun(5, func() { err = g.GenerateInto(context.Background(), out, ids, f.AudioMask, uid, ua) })
	if err != nil {
		t.Fatal(err)
	}
	if allocations != 0 {
		t.Fatalf("generation allocations %g", allocations)
	}
	for i, v := range out {
		if v != expected[i] {
			t.Fatal("not reproducible")
		}
	}
	if ids[0] != f.IDs[0] || ids[3] != f.IDs[3] {
		t.Fatal("prompt overwritten")
	}
}
func TestGenerationUpstreamGreedy(t *testing.T) {
	b, f := loadBackboneFixture(t)
	raw, err := os.ReadFile("../../testdata/omnivoice/backbone/generation.json")
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		Steps, Target    int
		Output, Schedule []int
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	uncond, err := NewBackbone(b.weights, ref.Target)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultGenerationConfig()
	cfg.Steps = ref.Steps
	cfg.PositionTemperature = 0
	g, err := NewGeneration(b, uncond, ref.Target, cfg)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]int, len(ref.Output))
	if err = g.GenerateInto(context.Background(), out, f.IDs, f.AudioMask, make([]int, 4), []bool{true, true}); err != nil {
		t.Fatal(err)
	}
	for i, v := range out {
		if v != ref.Output[i] {
			t.Fatalf("token %d got %d want %d", i, v, ref.Output[i])
		}
	}
	for i, v := range g.schedule {
		if v != ref.Schedule[i] {
			t.Fatal("schedule mismatch")
		}
	}
}

func TestGenerationCancellation(t *testing.T) {
	b, f := loadBackboneFixture(t)
	cfg := DefaultGenerationConfig()
	cfg.Guidance = 0
	g, err := NewGeneration(b, nil, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err = g.GenerateInto(ctx, make([]int, 4), f.IDs, f.AudioMask, nil, nil); err != context.Canceled {
		t.Fatal(err)
	}
}

func TestGenerationSharedWeightArena(t *testing.T) {
	w, err := loader.OpenWeights("../../testdata/omnivoice/backbone")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	c, err := NewBackbone(w, 4)
	if err != nil {
		t.Fatal(err)
	}
	u, err := NewBackboneSibling(c, 2)
	if err != nil {
		t.Fatal(err)
	}
	independent, err := NewBackbone(w, 2)
	if err != nil {
		t.Fatal(err)
	}
	if c.layer != u.layer || c.layer == independent.layer {
		t.Fatal("arena ownership incorrect")
	}
	cfg := DefaultGenerationConfig()
	cfg.Steps = 4
	shared, err := NewGeneration(c, u, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := NewGeneration(c, independent, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids := []int{2, 3, 4, 4, 2, 3, 4, 4}
	mask := []bool{false, false, true, true}
	ui := []int{4, 4, 4, 4}
	um := []bool{true, true}
	got, want := make([]int, 4), make([]int, 4)
	if err = baseline.GenerateInto(context.Background(), want, ids, mask, ui, um); err != nil {
		t.Fatal(err)
	}
	for j := 0; j < 3; j++ {
		if err = shared.GenerateInto(context.Background(), got, ids, mask, ui, um); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatal("shared arena changed generation")
		}
	}
	if n := testing.AllocsPerRun(3, func() {
		if err := shared.GenerateInto(context.Background(), got, ids, mask, ui, um); err != nil {
			panic(err)
		}
	}); n != 0 {
		t.Fatalf("allocations %g", n)
	}
	if _, err := NewBackboneSibling(nil, 2); err == nil {
		t.Fatal("nil parent accepted")
	}
}

func TestGenerationReconfigureParityAndZeroAllocs(t *testing.T) {
	w, err := loader.OpenWeights("../../testdata/omnivoice/backbone")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	cond, err := NewBackbone(w, 4)
	if err != nil {
		t.Fatal(err)
	}
	uncond, err := NewBackboneSibling(cond, 2)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultGenerationConfig()
	cfg.Steps = 4
	reused, err := NewGeneration(cond, uncond, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	cond4 := []int{2, 3, 4, 4, 2, 3, 4, 4}
	mask4 := []bool{false, false, true, true}
	uncond2 := []int{4, 4, 4, 4}
	umask2 := []bool{true, true}
	cond3 := sliceBookMajor(cond4, 2, 4, 3)
	mask3 := append([]bool(nil), mask4[:3]...)
	uncond1 := sliceBookMajor(uncond2, 2, 2, 1)
	umask1 := append([]bool(nil), umask2[:1]...)
	cases := []struct {
		condTokens   int
		uncondTokens int
		target       int
		condIDs      []int
		condAudio    []bool
		uncondIDs    []int
		uncondAudio  []bool
	}{
		{condTokens: 4, uncondTokens: 2, target: 2, condIDs: cond4, condAudio: mask4, uncondIDs: uncond2, uncondAudio: umask2},
		{condTokens: 3, uncondTokens: 1, target: 1, condIDs: cond3, condAudio: mask3, uncondIDs: uncond1, uncondAudio: umask1},
		{condTokens: 4, uncondTokens: 2, target: 2, condIDs: cond4, condAudio: mask4, uncondIDs: uncond2, uncondAudio: umask2},
	}
	for i, tc := range cases {
		indCond, err := NewBackbone(w, tc.condTokens)
		if err != nil {
			t.Fatal(err)
		}
		indUncond, err := NewBackboneSibling(indCond, tc.uncondTokens)
		if err != nil {
			t.Fatal(err)
		}
		independent, err := NewGeneration(indCond, indUncond, tc.target, cfg)
		if err != nil {
			t.Fatal(err)
		}
		want := make([]int, 2*tc.target)
		got := make([]int, len(want))
		condIDs := append([]int(nil), tc.condIDs...)
		uncondIDs := append([]int(nil), tc.uncondIDs...)
		if err = independent.GenerateInto(context.Background(), want, condIDs, tc.condAudio, uncondIDs, tc.uncondAudio); err != nil {
			t.Fatal(err)
		}
		condIDs = append([]int(nil), tc.condIDs...)
		uncondIDs = append([]int(nil), tc.uncondIDs...)
		if err = cond.Reconfigure(tc.condTokens); err != nil {
			t.Fatal(err)
		}
		if err = uncond.Reconfigure(tc.uncondTokens); err != nil {
			t.Fatal(err)
		}
		if err = reused.Reconfigure(tc.target); err != nil {
			t.Fatal(err)
		}
		if err = reused.GenerateInto(context.Background(), got, condIDs, tc.condAudio, uncondIDs, tc.uncondAudio); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("case %d got %v want %v", i, got, want)
		}
	}
	if err := reused.Reconfigure(0); err == nil {
		t.Fatal("zero target accepted")
	}
	if err := cond.Reconfigure(3); err != nil {
		t.Fatal(err)
	}
	if err := uncond.Reconfigure(1); err != nil {
		t.Fatal(err)
	}
	if err := reused.Reconfigure(2); err == nil {
		t.Fatal("target above active unconditional tokens accepted")
	}
	if err := cond.Reconfigure(4); err != nil {
		t.Fatal(err)
	}
	if err := uncond.Reconfigure(2); err != nil {
		t.Fatal(err)
	}
	workCond4 := make([]int, len(cond4))
	workUncond2 := make([]int, len(uncond2))
	workCond3 := make([]int, len(cond3))
	workUncond1 := make([]int, len(uncond1))
	out2 := make([]int, 4)
	out1 := make([]int, 2)
	var allocErr error
	allocs := testing.AllocsPerRun(10, func() {
		if allocErr = cond.Reconfigure(4); allocErr != nil {
			return
		}
		if allocErr = uncond.Reconfigure(2); allocErr != nil {
			return
		}
		if allocErr = reused.Reconfigure(2); allocErr != nil {
			return
		}
		copy(workCond4, cond4)
		copy(workUncond2, uncond2)
		allocErr = reused.GenerateInto(context.Background(), out2, workCond4, mask4, workUncond2, umask2)
		if allocErr != nil {
			return
		}
		if allocErr = cond.Reconfigure(3); allocErr != nil {
			return
		}
		if allocErr = uncond.Reconfigure(1); allocErr != nil {
			return
		}
		if allocErr = reused.Reconfigure(1); allocErr != nil {
			return
		}
		copy(workCond3, cond3)
		copy(workUncond1, uncond1)
		allocErr = reused.GenerateInto(context.Background(), out1, workCond3, mask3, workUncond1, umask1)
		if allocErr != nil {
			return
		}
		if allocErr = cond.Reconfigure(4); allocErr != nil {
			return
		}
		if allocErr = uncond.Reconfigure(2); allocErr != nil {
			return
		}
		if allocErr = reused.Reconfigure(2); allocErr != nil {
			return
		}
		copy(workCond4, cond4)
		copy(workUncond2, uncond2)
		allocErr = reused.GenerateInto(context.Background(), out2, workCond4, mask4, workUncond2, umask2)
	})
	if allocErr != nil {
		t.Fatal(allocErr)
	}
	if allocs != 0 {
		t.Fatalf("reuse allocs %g", allocs)
	}
}

func TestGenerationRejectsStaleBackboneShape(t *testing.T) {
	b, f := loadBackboneFixture(t)
	cfg := DefaultGenerationConfig()
	cfg.Guidance = 0
	g, err := NewGeneration(b, nil, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Reconfigure(1); err != nil {
		t.Fatal(err)
	}
	ids := sliceBookMajor(f.IDs, g.books, f.Tokens, 1)
	if err := g.GenerateInto(context.Background(), make([]int, g.books*2), ids, []bool{true}, nil, nil); err == nil {
		t.Fatal("accepted stale generation target larger than active backbone")
	}
	if err := g.Reconfigure(1); err != nil {
		t.Fatal(err)
	}
	if err := g.GenerateInto(context.Background(), make([]int, g.books), ids, []bool{true}, nil, nil); err != nil {
		t.Fatal(err)
	}
	var empty Generation
	if err := empty.Reconfigure(1); err == nil {
		t.Fatal("accepted uninitialised generation")
	}
}

func TestGenerationMidFlightCancelAndReuse(t *testing.T) {
	b, f := loadBackboneFixture(t)
	u, err := NewBackbone(b.weights, f.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	cfg := DefaultGenerationConfig()
	cfg.Steps = 4
	g, err := NewGeneration(b, u, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, ui := append([]int(nil), f.IDs...), append([]int(nil), f.IDs...)
	want, out := make([]int, 4), make([]int, 4)
	if err := g.GenerateInto(context.Background(), want, ids, f.AudioMask, ui, f.AudioMask); err != nil {
		t.Fatal(err)
	}
	for _, checks := range []int{3, 10, 20} {
		ctx := &cancelAfterChecks{Context: context.Background(), remaining: checks}
		if err := g.GenerateInto(ctx, out, ids, f.AudioMask, ui, f.AudioMask); err != context.Canceled {
			t.Fatalf("checks%d got%v", checks, err)
		}
		if err := g.GenerateInto(context.Background(), out, ids, f.AudioMask, ui, f.AudioMask); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(out, want) {
			t.Fatalf("retry checks%d got%v want%v", checks, out, want)
		}
	}
}
