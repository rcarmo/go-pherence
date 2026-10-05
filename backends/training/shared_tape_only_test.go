package training

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func TestTrainingSharedTapeOnlyPlanAndHostRejection(t *testing.T) {
	ctx := context.Background()
	fixture := residentFixture()
	generic, err := SharedResidentBytes(fixture)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := SharedTapeResidentBytes(fixture)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Bytes != generic.Bytes-4*int64(generic.MaxElements) || plan.Elements != generic.Elements || plan.Parameters != generic.Parameters {
		t.Fatal("tape-only payload", plan, generic)
	}
	d := &sharedHostDriver{}
	if s, err := NewTapeOnlySharedWorkspaceResidentState(ctx, d, fixture, plan.Bytes-1); err == nil || s != nil || len(d.buffers) != 0 {
		t.Fatal("underbudget allocated")
	}
	s, err := NewTapeOnlySharedWorkspaceResidentState(ctx, d, fixture, plan.Bytes)
	if err != nil {
		t.Fatal(err)
	}
	if s.shared.branch != nil || !s.shared.tapeOnly || len(s.owned) != 5*len(fixture.Parameters)+6 {
		t.Fatal("unexpected branch/owner inventory")
	}
	var bytes int64
	for _, b := range d.buffers {
		bytes += 4 * int64(len(b.values))
	}
	if bytes != plan.Bytes {
		t.Fatal("allocation ledger", bytes, plan.Bytes)
	}
	before, err := s.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	events := map[string]int{}
	for name, n := range d.events {
		events[name] = n
	}
	if err := s.Accumulate(ctx, map[string][]float32{"a": {.1, .2}}, 1); err == nil {
		t.Fatal("host branch accepted")
	}
	if err := s.sharedAccumulate(ctx, []string{"a"}, nil, nil, nil, map[string][]float32{"a": {.1, .2}}, 1); err == nil {
		t.Fatal("internal host bypass")
	}
	if !reflect.DeepEqual(events, d.events) || s.dirty || s.terminal != nil || s.borrowed || s.shared.active {
		t.Fatal("host rejection mutated state")
	}
	after, err := s.Snapshot(ctx)
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatal("host rejection changed snapshot", err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := SharedTapeResidentBytes(ResidentSnapshot{}); err == nil {
		t.Fatal("invalid plan")
	}
}

func TestTrainingSharedTapeOnlyDenseCompactAndSnapshot(t *testing.T) {
	for _, compact := range []bool{false, true} {
		t.Run(map[bool]string{false: "dense", true: "compact"}[compact], func(t *testing.T) {
			ctx := context.Background()
			fixture := residentFixture()
			if compact {
				fixture = sharedCompactFixture()
			}
			d := &sharedHostDriver{}
			s, err := NewTapeOnlySharedWorkspaceResidentState(ctx, d, fixture, 4096)
			if err != nil {
				t.Fatal(err)
			}
			legacy := &sharedHostDriver{}
			other, err := NewSharedWorkspaceResidentState(ctx, legacy, fixture, 4096)
			if err != nil {
				t.Fatal(err)
			}
			for _, state := range []*ResidentState{s, other} {
				for _, scale := range []float32{0, .25, 1} {
					if compact {
						tape, b, c := compactBindingFixture()
						_, err = state.RunCompactZeroPrunedTape(ctx, tape, b, c, []int{6}, true, scale, 4096)
					} else {
						tape, b := residentBindingFixture()
						_, err = state.RunPrunedTape(ctx, tape, b, []int{2}, true, scale, 1024)
					}
					if err != nil {
						t.Fatal(err)
					}
				}
			}
			for i, e := range s.entries {
				got, want := e.gradient.(*residentHostBuffer).values, other.entries[i].gradient.(*residentHostBuffer).values
				for j := range got {
					if math.Abs(float64(got[j])-float64(want[j])) > 2e-6 {
						t.Fatal("tape-only changed accumulation")
					}
				}
			}
			if compact {
				if _, err := s.Update(ctx, 1); err != nil {
					t.Fatal(err)
				}
				if _, err := other.Update(ctx, 1); err != nil {
					t.Fatal(err)
				}
				a, err := s.Snapshot(ctx)
				if err != nil {
					t.Fatal(err)
				}
				b, err := other.Snapshot(ctx)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(a, b) {
					t.Fatal("optimizer state differs")
				}
			}
			if err := s.Close(); err != nil {
				t.Fatal(err)
			}
			if err := other.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestTrainingSharedTapeOnlyTerminalFailure(t *testing.T) {
	for _, event := range []string{"new", "inventory-admit", "upload", "tape-run", "copy"} {
		t.Run(event, func(t *testing.T) {
			d := &sharedHostDriver{}
			d.sentinel = errors.New("injected tape-only failure")
			if event == "new" || event == "inventory-admit" || event == "upload" {
				d.fail = event
				d.at = 1
			}
			s, err := NewTapeOnlySharedWorkspaceResidentState(context.Background(), d, residentFixture(), 4096)
			if err == nil {
				d.fail = event
				d.at = 1
				tape, b := residentBindingFixture()
				_, err = s.RunPrunedTape(context.Background(), tape, b, []int{2}, true, 1, 1024)
			}
			if err == nil || s == nil || s.terminal == nil || s.Close() == nil {
				t.Fatal("uncertain owner not terminal", event, err)
			}
			if _, err := s.Snapshot(context.Background()); err == nil {
				t.Fatal("failed owner snapshot")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	d := &sharedHostDriver{}
	if s, err := NewTapeOnlySharedWorkspaceResidentState(ctx, d, residentFixture(), 4096); !errors.Is(err, context.Canceled) || s != nil || len(d.buffers) > 0 {
		t.Fatal("cancel allocated", err)
	}
}
