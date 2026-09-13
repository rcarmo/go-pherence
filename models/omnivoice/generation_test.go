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
