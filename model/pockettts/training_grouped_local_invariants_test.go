package pockettts

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"testing"
)

func newPocketGroup(tape train.Tape, b map[string]train.ResidentTapeBinding) *pocketGroupLocal {
	s := &pocketGroupState{p: map[string][]float32{}, g: map[string][]float32{}, ready: map[string]bool{}}
	for n, v := range b {
		s.p[n] = append([]float32(nil), tape.Tensors[v.Parameter].Values...)
		s.g[n] = make([]float32, len(s.p[n]))
	}
	return &pocketGroupLocal{state: s, faultItem: -1}
}
func pocketGroupTiny() (train.Tape, map[string]train.ResidentTapeBinding, []int, pocketGroupPolicy) {
	// A.G reads B.P even though B is selected by another group. B.G reads A.P.
	t := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{2}}, {Shape: []int{1}, Values: []float32{3}}, {Shape: []int{1}}, {Shape: []int{1}}, {Shape: []int{1}, Values: []float32{.75}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 2}, {Op: "vector", Inputs: []int{1, 0}, Output: 3, Scale: 3}}, Outputs: []int{2, 3, 4}}
	return t, map[string]train.ResidentTapeBinding{"a": {Parameter: 0, Gradient: 2}, "b": {Parameter: 1, Gradient: 3}}, []int{4}, pocketGroupPolicy{kind: "component", roles: []string{"metric"}, extents: []int{1}}
}
func TestPocketGroupedLocalFreezeCrossP(t *testing.T) {
	tape, b, r, policy := pocketGroupTiny()
	p := newPocketGroup(tape, b)
	groups := [][]string{{"a"}, {"b"}}
	pa, pb := &p.state.g["a"][0], &p.state.g["b"][0]
	p.callback = func(stage string, gi, item int) {
		if stage == "plan" {
			p.state.p["b"][0] = 900
			tape.Tensors[1].Values[0] = 800
			b["a"] = b["b"]
			groups[1][0] = "a"
			r[0] = 3
			policy.extents[0] = 9
			tape.Commands[1].Scale = 900
		}
	}
	out, e := p.run(context.Background(), tape, b, nil, r, groups, .25, policy)
	if e != nil {
		t.Fatal(e)
	}
	if p.state.g["a"][0] != 2 || p.state.g["b"][0] != 2.25 || pa != &p.state.g["a"][0] || pb != &p.state.g["b"][0] {
		t.Fatal("frozen cross-P/stable G", p.state.g)
	}
	if out[0][0] != .75 || p.live != 0 || p.ledger.MaxLive != 1 || p.ledger.Closed != 2 {
		t.Fatal("publication/lifetime", p.ledger)
	}
	out[0][0] = 7
	if e = p.state.probe(context.Background(), "snapshot"); e != nil {
		t.Fatal(e)
	}
	t.Logf("crossP hostledger=%+v", p.ledger)
}
func TestPocketGroupedLocalPreflight(t *testing.T) {
	for _, kind := range []string{"empty", "missing", "duplicate", "foreign", "group-size", "tuple", "results", "outputs", "shape", "operands", "metadata-beforefreeze", "metadata-late", "plan-late", "effective-P", "invalid-original", "bad-policy", "compact-provenance", "late-duplicate"} {
		t.Run(kind, func(t *testing.T) {
			tape, b, r, policy := pocketGroupTiny()
			p := newPocketGroup(tape, b)
			groups := [][]string{{"a"}, {"b"}}
			compact := map[string]train.ResidentCompactEmbeddingBinding(nil)
			switch kind {
			case "empty":
				groups = nil
			case "missing":
				groups = groups[:1]
			case "duplicate":
				groups[1][0] = "a"
			case "foreign":
				groups[1][0] = "z"
			case "group-size":
				groups[0] = make([]string, 9)
			case "tuple":
				tape.Commands[0].Mode = 99
			case "results":
				r = []int{4, 4}
			case "outputs":
				tape.Outputs = make([]int, 4097)
			case "shape":
				tape.Tensors[0].Shape = []int{1, 1, 1, 1}
			case "operands":
				tape.Commands[0].Inputs = make([]int, 17)
			case "metadata-beforefreeze":
				p.metadataLimit = 1
			case "metadata-late":
				p.metadataLimit = 5400
			case "plan-late":
				p.fault = "plan"
				p.faultGroup = 1
			case "effective-P":
				p.state.p["a"][0] = float32(math.NaN())
			case "invalid-original":
				tape.Tensors[0].Values[0] = float32(math.NaN())
			case "bad-policy":
				policy.extents[0] = 2
			case "compact-provenance":
				compact = map[string]train.ResidentCompactEmbeddingBinding{"a": {Index: 4, Rows: 1, Width: 1, Vocabulary: 1}}
			case "late-duplicate":
				groups = append(groups, groups[0])
			}
			before := map[string][]float32{"a": append([]float32(nil), p.state.g["a"]...), "b": append([]float32(nil), p.state.g["b"]...)}
			out, e := p.run(context.Background(), tape, b, compact, r, groups, 1, policy)
			if e == nil || out != nil || p.ledger.Constructed != 0 || len(p.ledger.Groups) != 0 || p.state.terminal != nil || !reflect.DeepEqual(before, p.state.g) {
				t.Fatal("preflight executed/changed G", kind, e, p.ledger)
			}
			if kind == "metadata-beforefreeze" && p.ledger.FreezeCopies != 0 {
				t.Fatal("copy before initial reservation")
			}
			if kind == "metadata-late" && p.ledger.Plans != 1 {
				t.Fatal("must reject reservation after one admitted plan", p.ledger)
			}
			t.Logf("reject %s: %v plans=%d freezes=%d reserved=%d", kind, e, p.ledger.Plans, p.ledger.FreezeCopies, p.ledger.MetadataReservation)
		})
	}
}
func TestPocketGroupedLocalFailureFences(t *testing.T) {
	for _, stage := range []string{"construct", "nil", "constructed", "run", "witness", "download", "lookup", "actual", "copy", "close", "closed", "publish"} {
		for _, gi := range []int{0, 1} {
			t.Run(stage+string(rune('0'+gi)), func(t *testing.T) {
				tape, b, r, policy := pocketGroupTiny()
				p := newPocketGroup(tape, b)
				p.fault = stage
				p.faultGroup = gi
				p.nilOwner = stage == "nil"
				if stage == "publish" {
					p.faultGroup = 1
				}
				out, e := p.run(context.Background(), tape, b, nil, r, [][]string{{"a"}, {"b"}}, 1, policy)
				if e == nil || out != nil {
					t.Fatal("fault escaped", stage, e)
				}
				wantTerminal := !(stage == "nil" && gi == 0)
				if (p.state.terminal != nil) != wantTerminal {
					t.Fatal("terminal state", stage, gi, e)
				}
				if gi == 1 && p.state.g["a"][0] != 8 {
					t.Fatal("lost first commit")
				}
				if stage == "closed" || stage == "publish" {
					if len(p.state.retained) != 0 || p.live != 0 {
						t.Fatal("resurrected closed owner")
					}
				} else if wantTerminal && stage != "nil" {
					if len(p.state.retained) != 1 || p.state.retained[0].closed {
						t.Fatal("uncertain owner not retained once")
					}
				}
				if wantTerminal {
					for _, op := range []string{"update", "snapshot", "close", "accumulate"} {
						if p.state.probe(context.Background(), op) == nil {
							t.Fatal("terminal allowed", op)
						}
					}
				}
			})
		}
	}
}
func TestPocketGroupedLocalCancellation(t *testing.T) {
	for _, stage := range []string{"construct", "constructed", "run", "dispatch", "dispatched", "witness", "download", "lookup", "actual", "copy", "close", "closed", "publish"} {
		t.Run(stage, func(t *testing.T) {
			tape, b, r, policy := pocketGroupTiny()
			p := newPocketGroup(tape, b)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			p.callback = func(s string, gi, item int) {
				if s == stage && gi == 1 {
					cancel()
				}
			}
			out, e := p.run(ctx, tape, b, nil, r, [][]string{{"a"}, {"b"}}, 1, policy)
			if e == nil || out != nil || p.state.terminal == nil || p.state.g["a"][0] != 8 {
				t.Fatal("late cancellation", stage, e)
			}
			if stage == "construct" || stage == "closed" || stage == "publish" {
				if len(p.state.retained) != 0 || p.live != 0 {
					t.Fatal("closed/nil owner retained", stage)
				}
			} else if len(p.state.retained) != 1 || p.state.retained[0].closed {
				t.Fatal("lost uncertain owner", stage)
			}
		})
	}
}
func TestPocketGroupedLocalPartialCopyAndWitness(t *testing.T) {
	for _, kind := range []string{"partial-copy", "nonfinite"} {
		t.Run(kind, func(t *testing.T) {
			tape, b, r, policy := pocketGroupTiny()
			p := newPocketGroup(tape, b)
			if kind == "partial-copy" {
				p.fault = "copy"
				p.faultItem = 1
			} else {
				p.corrupt = func(gi int, v [][]float32) { v[0][0] = float32(math.NaN()) }
			}
			out, e := p.run(context.Background(), tape, b, nil, r, [][]string{{"a", "b"}}, 1, policy)
			if e == nil || out != nil || p.state.terminal == nil || len(p.state.retained) != 1 {
				t.Fatal(e)
			}
			if kind == "partial-copy" {
				if p.state.g["a"][0] != 8 || p.state.g["b"][0] != 0 || !p.state.ready["a"] || p.state.ready["b"] {
					t.Fatal("partial commit", p.state.g)
				}
			} else if p.state.ready["a"] || p.state.ready["b"] {
				t.Fatal("nonfinite before G")
			}
		})
	}
}
func TestPocketGroupedLocalReentry(t *testing.T) {
	for _, stage := range []string{"plan", "construct", "constructed", "run", "dispatch", "witness", "download", "lookup", "actual", "copy", "close", "closed", "publish"} {
		t.Run(stage, func(t *testing.T) {
			tape, b, r, policy := pocketGroupTiny()
			p := newPocketGroup(tape, b)
			calls := 0
			p.callback = func(s string, gi, item int) {
				if s != stage {
					return
				}
				calls++
				if out, e := p.run(context.Background(), tape, b, nil, r, [][]string{{"a"}, {"b"}}, 1, policy); e == nil || out != nil {
					t.Fatal("nested work")
				}
				for _, op := range []string{"update", "snapshot", "close", "accumulate"} {
					if p.state.probe(context.Background(), op) == nil {
						t.Fatal("active fence", op)
					}
				}
			}
			out, e := p.run(context.Background(), tape, b, nil, r, [][]string{{"a"}, {"b"}}, 1, policy)
			if e != nil || out == nil || calls == 0 {
				t.Fatal(e, calls)
			}
		})
	}
}

func TestPocketGroupedLocalMalformedAndLifetime(t *testing.T) {
	for _, kind := range []string{"nil-coordinator", "nil-state", "nil-context", "cancel-entry", "closed-state", "terminal-state", "nonfinite-scale", "ceiling-override", "G-alias", "policy-count", "policy-kind", "nil-ready", "typed-short-output", "typed-mask", "typed-latent", "effective-gather", "invalid-original-gather"} {
		t.Run(kind, func(t *testing.T) {
			tape, b, r, policy := pocketGroupTiny()
			p := newPocketGroup(tape, b)
			ctx := context.Background()
			scale := float32(1)
			switch kind {
			case "nil-coordinator":
				p = nil
			case "nil-state":
				p.state = nil
			case "nil-context":
				ctx = nil
			case "cancel-entry":
				x, c := context.WithCancel(ctx)
				c()
				ctx = x
			case "closed-state":
				p.state.closed = true
			case "terminal-state":
				p.state.terminal = fmt.Errorf("prior fault")
			case "nonfinite-scale":
				scale = float32(math.NaN())
			case "ceiling-override":
				p.metadataLimit = pocketGroupLimit + 1
			case "G-alias":
				p.state.g["b"] = p.state.g["a"]
			case "nil-ready":
				p.state.ready = nil
			case "policy-count":
				policy.roles = nil
			case "policy-kind":
				policy.kind = "foreign"
			case "typed-short-output", "typed-mask", "typed-latent":
				later, rr, pp := pocketGroupLaterFixture(t, 0, false)
				tape, b, r, policy = later.Tape, later.ResidentBindings, rr, pp
				p = newPocketGroup(tape, b)
				switch kind {
				case "typed-short-output":
					tape.Outputs = tape.Outputs[:5]
				case "typed-mask":
					policy.step.in.Mask = nil
				case "typed-latent":
					policy.step.in.Latent = -1
				}
			case "effective-gather", "invalid-original-gather":
				tape.Tensors = append(tape.Tensors, train.TensorSpec{Shape: []int{1}})
				tape.Commands = append(tape.Commands, train.Command{Op: "embedding", Mode: 0, Inputs: []int{1, 0}, Output: 5, Rows: 1, Width: 1, Parts: 1})
				tape.Tensors[0].Values[0] = 0
				p.state.p["a"][0] = 1
				if kind == "invalid-original-gather" {
					tape.Tensors[0].Values[0] = 1
					p.state.p["a"][0] = 0
				}
			}
			out, e := p.run(ctx, tape, b, nil, r, [][]string{{"a"}, {"b"}}, scale, policy)
			if e == nil || out != nil {
				t.Fatal("malformed accepted", kind)
			}
			if p != nil && p.ledger.Constructed != 0 {
				t.Fatal("malformed executed", kind)
			}
		})
	}
	tape, b, r, policy := pocketGroupTiny()
	p := newPocketGroup(tape, b)
	out, e := p.run(context.Background(), tape, b, nil, r, [][]string{{"a"}, {"b"}}, 1, policy)
	if e != nil {
		t.Fatal(e)
	}
	before := append([]float32(nil), out[0]...)
	tape.Tensors[4].Values[0] = 9
	p.state.p["a"][0] = 7
	if e = p.state.probe(context.Background(), "close"); e != nil {
		t.Fatal(e)
	}
	if !reflect.DeepEqual(out[0], before) {
		t.Fatal("published output lifetime")
	}
}

func TestPocketGroupedLocalCompactPrior(t *testing.T) {
	ix, e := train.BuildCompactEmbeddingIndex([]float32{0, 2, 0}, 4)
	if e != nil {
		t.Fatal(e)
	}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{8}, Values: make([]float32, 8)}, {Shape: []int{1}, Values: []float32{2}}, {Shape: []int{6}, Values: []float32{1, 2, 3, 4, 5, 6}}, {Shape: []int{len(ix)}, Values: ix}, {Shape: []int{6}}, {Shape: []int{1}}, {Shape: []int{1}, Values: []float32{.5}}}, Commands: []train.Command{{Op: "embedding", Mode: 3, Rows: 3, Width: 2, Parts: 4, Inputs: []int{2, 3}, Output: 4}, {Op: "vector", Inputs: []int{1, 1}, Output: 5, Scale: 2}}, Outputs: []int{4, 5, 3, 6}}
	b := map[string]train.ResidentTapeBinding{"a": {Parameter: 0, Gradient: 4}, "b": {Parameter: 1, Gradient: 5}}
	c := map[string]train.ResidentCompactEmbeddingBinding{"a": {Index: 3, Rows: 3, Width: 2, Vocabulary: 4}}
	policy := pocketGroupPolicy{kind: "component", roles: []string{"metric"}, extents: []int{1}}
	var stats pocketAdapterErrors
	for _, order := range []int{0, 1} {
		for _, scale := range []float32{0, .25, 1} {
			p := newPocketGroup(tape, b)
			prior := pocketGroupPrior(p)
			ptr := &p.state.g["a"][0]
			groups := [][]string{{"a"}, {"b"}}
			if order == 1 {
				groups = [][]string{{"b"}, {"a"}}
			}
			out, e := p.run(context.Background(), tape, b, c, []int{6}, groups, scale, policy)
			if e != nil {
				t.Fatal(e)
			}
			if ptr != &p.state.g["a"][0] || out[0][0] != .5 {
				t.Fatal("compact stable G/result")
			}
			pocketGroupCompareG(t, "independent compact", p, map[string][]float32{"a": {6, 8, 0, 0, 3, 4, 0, 0}, "b": {6}}, prior, scale, 2e-6, &stats)
		}
	}
	stats.log(t, "grouped compact independent F64 prior+scale G")
}

func TestPocketGroupedLocalWorkAndOverflow(t *testing.T) {
	t.Run("large-work-metadata-binding", func(t *testing.T) {
		// 128 distinct scalar P/G names; each G depends on the same 513-command
		// chain. The current retained-plan metadata ceiling binds before the
		// 65536 work limit; record that exact constraint without widening it.
		tape := train.Tape{}
		b := map[string]train.ResidentTapeBinding{}
		groups := [][]string{}
		for i := 0; i < 128; i++ {
			id := len(tape.Tensors)
			tape.Tensors = append(tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{1}})
			name := fmt.Sprintf("p%03d", i)
			b[name] = train.ResidentTapeBinding{Parameter: id}
			groups = append(groups, []string{name})
		}
		last := 0
		for i := 0; i < 513; i++ {
			id := len(tape.Tensors)
			tape.Tensors = append(tape.Tensors, train.TensorSpec{Shape: []int{1}})
			tape.Commands = append(tape.Commands, train.Command{Op: "vector", Inputs: []int{last, 0}, Output: id, Scale: 0})
			last = id
		}
		metric := len(tape.Tensors)
		tape.Tensors = append(tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{0}})
		tape.Outputs = []int{metric}
		for i := 0; i < 128; i++ {
			name := fmt.Sprintf("p%03d", i)
			id := len(tape.Tensors)
			tape.Tensors = append(tape.Tensors, train.TensorSpec{Shape: []int{1}})
			tape.Commands = append(tape.Commands, train.Command{Op: "vector", Inputs: []int{last, 0}, Output: id, Scale: 0})
			v := b[name]
			v.Gradient = id
			b[name] = v
			tape.Outputs = append(tape.Outputs, id)
		}
		p := newPocketGroup(tape, b)
		out, e := p.run(context.Background(), tape, b, nil, []int{metric}, groups, 1, pocketGroupPolicy{kind: "component", roles: []string{"metric"}, extents: []int{1}})
		if e == nil || out != nil || p.ledger.Constructed != 0 || len(p.ledger.Groups) != 0 || p.ledger.Commands > 65536 {
			t.Fatal("large-work binding guard", e)
		}
		t.Logf("checked bound: %v commands=%d metadata=%d plans=%d", e, p.ledger.Commands, p.ledger.MetadataReservation, p.ledger.Plans)
	})
	t.Run("accumulate-overflow", func(t *testing.T) {
		tape, b, r, policy := pocketGroupTiny()
		p := newPocketGroup(tape, b)
		out, e := p.run(context.Background(), tape, b, nil, r, [][]string{{"a"}, {"b"}}, math.MaxFloat32, policy)
		if e == nil || out != nil || p.state.terminal == nil || len(p.state.retained) != 1 || p.state.ready["a"] {
			t.Fatal("overflow fence", e)
		}
	})
}
