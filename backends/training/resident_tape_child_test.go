package training

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

func childFixture() (Tape, []int) {
	// Root2 -> input1, current parameter a ->input2, output3. Dead leading
	// constant forces stable nonidentity remaps; output3 represents prediction.
	t, _, _ := leaseConsumerFixture()
	return t, []int{3}
}
func beginChildFixture(t *testing.T, limit int64) (*ResidentState, *leaseHostDriver, *ResidentTapeLease, *ResidentTapeLease) {
	t.Helper()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	root, err := s.BeginPrunedTapeLease(context.Background(), p, b, []int{2}, limit)
	if err != nil {
		t.Fatal(err)
	}
	source, outputs := childFixture()
	_, bindings, _ := leaseConsumerFixture()
	child, err := root.BeginChildTape(context.Background(), source, bindings, outputs, []ResidentLeaseEdge{{Source: root, Output: 2, Input: 1}})
	if err != nil {
		t.Fatal(err)
	}
	return s, d, root, child
}
func multiConsumerFixture() (Tape, map[string]ResidentTapeBinding, []int) {
	// Reads condition(root) and prediction(child) without download/re-upload.
	t := Tape{Tensors: []TensorSpec{{Shape: []int{1}, Values: []float32{7}}, {Shape: []int{2}, Values: []float32{99, 99}}, {Shape: []int{2}, Values: []float32{88, 88}}, {Shape: []int{2}}, {Shape: []int{2}, Values: []float32{66, 66}}, {Shape: []int{2}, Values: []float32{.4, .2}}}, Commands: []Command{{Op: "vector", Inputs: []int{1, 2}, Output: 3, Scale: .5}}, Outputs: []int{3, 5}}
	return t, map[string]ResidentTapeBinding{"a": {Parameter: 4, Gradient: 5}}, []int{3}
}
func TestTrainingResidentTapeChildLifetime(t *testing.T) {
	ctx := context.Background()
	s, d, root, child := beginChildFixture(t, 1024)
	if len(root.live) != 2 || root.liveBytes != 60 || d.executions != 2 || d.downloads != 0 || s.dirty {
		t.Fatal("retained child accounting")
	}
	if root.Close() == nil || s.terminal != nil || d.owners[0].closed {
		t.Fatal("parent prematurely freed/poisoned")
	}
	source, pinned := childFixture()
	_, bindings, _ := leaseConsumerFixture()
	sibling, err := root.BeginChildTape(ctx, source, bindings, pinned, []ResidentLeaseEdge{{Source: child, Output: 3, Input: 1}})
	if err != nil {
		t.Fatal(err)
	}
	if child.Close() == nil || s.terminal != nil {
		t.Fatal("sibling borrower did not pin source")
	}
	tape, b, outputs := multiConsumerFixture()
	edges := []ResidentLeaseEdge{{Source: root, Output: 2, Input: 1}, {Source: child, Output: 3, Input: 2}}
	values, err := sibling.RunSourcesTape(ctx, tape, b, nil, outputs, edges, true, .5)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float64{2.25, 1.25} {
		if math.Abs(float64(values[0][i])-want) > 2e-6 {
			t.Fatal("multi-source value", values)
		}
	}
	// Only the final consumer explicitly accumulates; child/root forward never do.
	if d.accumulations != 1 || d.downloads != 1 || d.executions != 4 || d.owners[0].closed || d.owners[1].closed {
		t.Fatal("implicit forward gradients/download")
	}
	if sibling.Close() != nil || child.Close() != nil || root.Close() != nil || s.lease != nil {
		t.Fatal("reverse close")
	}
	if !reflect.DeepEqual(d.closeOrder, []int{4, 3, 2, 1}) || root.liveBytes != 0 || len(root.live) != 0 {
		t.Fatal("close order/live accounting", d.closeOrder)
	}
	if child.Close() != nil || s.Close() != nil {
		t.Fatal("idempotent close")
	}
}
func TestTrainingResidentTapeChildBudget(t *testing.T) {
	for _, limit := range []int64{35, 36} {
		t.Run(map[int64]string{35: "reject", 36: "exact"}[limit], func(t *testing.T) {
			s, d := requireLeaseResident(t)
			p := Tape{Tensors: []TensorSpec{{Shape: []int{1}, Values: []float32{1}}}, Outputs: []int{0}}
			root, err := s.BeginPrunedTapeLease(context.Background(), p, nil, []int{0}, limit)
			if err != nil {
				t.Fatal(err)
			}
			childTape := Tape{Tensors: []TensorSpec{{Shape: []int{1}, Values: []float32{99}}, {Shape: []int{1}, Values: []float32{2}}, {Shape: []int{1}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
			child, err := root.BeginChildTape(context.Background(), childTape, nil, []int{2}, []ResidentLeaseEdge{{Source: root, Output: 0, Input: 0}})
			if err != nil {
				t.Fatal(err)
			}
			sibling, err := root.BeginChildTape(context.Background(), childTape, nil, []int{2}, []ResidentLeaseEdge{{Source: child, Output: 2, Input: 0}})
			if err != nil {
				t.Fatal(err)
			}
			// Complete sources 4+12+12=28; utility consumer 8 -> exact36 ceiling.
			utility := Tape{Tensors: []TensorSpec{{Shape: []int{1}, Values: []float32{0}}, {Shape: []int{1}, Values: []float32{0}}}, Outputs: []int{0}}
			before := d.constructions
			_, err = sibling.RunSourcesTape(context.Background(), utility, nil, nil, []int{0}, nil, false, 1)
			if limit == 35 {
				if err == nil || d.constructions != before || s.terminal == nil {
					t.Fatal("all live sources not charged")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if sibling.Close() != nil {
				t.Fatal("close sibling")
			}
			if root.liveBytes != 16 {
				t.Fatal("released child charge")
			}
			if child.Close() != nil || root.Close() != nil || s.Close() != nil {
				t.Fatal("close")
			}
		})
	}
}
func TestTrainingResidentTapeChildAdmission(t *testing.T) {
	for _, kind := range []string{"source", "inventory", "budget", "output", "duplicate", "unbound", "foreign", "copy", "closed", "target", "extent", "conflict", "alias", "preflight", "cancel"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			s, d, root, child := beginChildFixture(t, 1024)
			tape, bindings, outputs := multiConsumerFixture()
			edges := []ResidentLeaseEdge{{Source: root, Output: 2, Input: 1}, {Source: child, Output: 3, Input: 2}}
			switch kind {
			case "source":
				tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "inventory":
				bindings["unknown"] = bindings["a"]
				delete(bindings, "a")
			case "budget":
				root.maxBytes = 99
				child.maxBytes = 99
			case "output":
				outputs = []int{0}
			case "duplicate":
				outputs = []int{3, 3}
			case "unbound":
				bindings = map[string]ResidentTapeBinding{}
			case "foreign":
				_, _, other, _ := beginChildFixture(t, 1024)
				edges[0].Source = other
			case "copy":
				copyOwner := *root
				edges[0].Source = &copyOwner
			case "closed":
				if child.Close() != nil {
					t.Fatal("close")
				}
			case "target":
				edges[0].Input = 3
			case "extent":
				edges[0].Input = 0
			case "conflict":
				edges[0].Input = 4
			case "alias":
				edges[1].Source = root
				edges[1].Output = 2
			case "preflight":
				d.preflightFailure = true
			case "cancel":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			before := d.constructions
			_, err := root.BeginChildTape(ctx, tape, bindings, outputs, edges)
			if err == nil || d.constructions != before || s.terminal == nil || root.Close() == nil || d.owners[0].closed {
				t.Fatal("child admission retention", kind, err)
			}
			assertLeaseBlocked(t, s)
		})
	}
}
func TestTrainingResidentTapeChildFailures(t *testing.T) {
	for _, kind := range []string{"construction", "nil", "run", "output", "download", "close"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			s, d := requireLeaseResident(t)
			p, b := leaseProducerFixture()
			root, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
			if err != nil {
				t.Fatal(err)
			}
			tape, outputs := childFixture()
			_, bindings, _ := leaseConsumerFixture()
			d.tapeFailure = kind
			child, err := root.BeginChildTape(ctx, tape, bindings, outputs, []ResidentLeaseEdge{{Source: root, Output: 2, Input: 1}})
			if kind == "output" {
				if child != nil || err == nil {
					t.Fatal("source output before allocation")
				}
			} else {
				if child == nil {
					t.Fatal("partial child discarded")
				}
				switch kind {
				case "download":
					if err != nil {
						t.Fatal(err)
					}
					_, err = child.DownloadOutputs(ctx, []int{3})
				case "close":
					if err != nil {
						t.Fatal(err)
					}
					err = child.Close()
				}
				if err == nil || len(root.live) != 2 || d.owners[len(d.owners)-1].closed {
					t.Fatal("child failure cleanup", kind, err)
				}
			}
			if s.terminal == nil || root.Close() == nil || d.owners[0].closed {
				t.Fatal("tree not retained")
			}
			assertLeaseBlocked(t, s)
		})
	}
}
func TestTrainingResidentTapeChildConsumerFailure(t *testing.T) {
	ctx := context.Background()
	s, d, root, child := beginChildFixture(t, 1024)
	tape, b, outputs := multiConsumerFixture()
	edges := []ResidentLeaseEdge{{Source: root, Output: 2, Input: 1}, {Source: child, Output: 3, Input: 2}}
	if _, err := child.RunSourcesTape(ctx, tape, b, nil, outputs, edges, true, .5); err != nil {
		t.Fatal(err)
	}
	gradient := append([]float32(nil), s.entries[s.index["a"]].gradient.(*residentHostBuffer).values...)
	d.fail = "accumulate"
	d.at = 2
	d.sentinel = errors.New("child consumer accumulation")
	if _, err := child.RunSourcesTape(ctx, tape, b, nil, outputs, edges, true, .5); err == nil {
		t.Fatal("failure")
	}
	if len(s.retained) != 1 || s.terminal == nil || d.owners[0].closed || d.owners[1].closed || d.owners[3].closed || !reflect.DeepEqual(gradient, s.entries[s.index["a"]].gradient.(*residentHostBuffer).values) {
		t.Fatal("failed multi-owner consumer freed/rolled back")
	}
	if child.Close() == nil || root.Close() == nil {
		t.Fatal("terminal close")
	}
	before := d.executions
	if _, err := root.RunSourcesTape(ctx, tape, b, nil, outputs, edges, true, .5); err == nil || d.executions != before {
		t.Fatal("retry")
	}
}
func TestTrainingResidentTapeChildF64(t *testing.T) {
	ctx := context.Background()
	s, d, root, child := beginChildFixture(t, 1024)
	tape, b, outputs := multiConsumerFixture()
	edges := []ResidentLeaseEdge{{Source: root, Output: 2, Input: 1}, {Source: child, Output: 3, Input: 2}}
	var count, sum, squares, maximum float64
	for trial := 0; trial < 64; trial++ {
		scale := float32(trial) * .01
		tape.Commands[0].Scale = scale
		values, err := child.RunSourcesTape(ctx, tape, b, nil, outputs, edges, true, .5)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range values[0] {
			parameter := float64(s.entries[s.index["a"]].parameter.(*residentHostBuffer).values[i])
			rootWant := parameter + float64([]float32{1, 2}[i])
			childWant := rootWant + 2*parameter
			want := rootWant + float64(scale)*childWant
			error := math.Abs(float64(v) - want)
			if error > 2e-6 {
				t.Fatal("F64 contract", error)
			}
			count++
			sum += error
			squares += error * error
			maximum = math.Max(maximum, error)
		}
	}
	if d.downloads != 64 || d.executions != 66 || d.accumulations != 64 {
		t.Fatal("only consumer downloads")
	}
	if child.Close() != nil || root.Close() != nil || s.Close() != nil {
		t.Fatal("close")
	}
	t.Logf("F64 comparisons %.0f mean_abs %.9g RMS %.9g max_abs %.9g unchanged absolute gate 2e-6", count, sum/count, math.Sqrt(squares/count), maximum)
}

func TestTrainingResidentTapeChildReentryAndCopy(t *testing.T) {
	ctx := context.Background()
	s, d, root, child := beginChildFixture(t, 1024)
	clone := *child
	if clone.admit(ctx, s) == nil {
		t.Fatal("copied child authority")
	}
	source, outputs := childFixture()
	_, bindings, _ := leaseConsumerFixture()
	if _, err := clone.BeginChildTape(ctx, source, bindings, outputs, nil); err == nil || s.terminal != nil {
		t.Fatal("copied child constructor authority")
	}
	d.onRun = func() {
		assertLeaseBlocked(t, s)
		if child.Close() == nil || root.Close() == nil {
			t.Fatal("reentrant producer release")
		}
		if _, err := child.DownloadOutputs(ctx, []int{3}); err == nil {
			t.Fatal("reentrant child download")
		}
		if _, err := root.BeginChildTape(ctx, source, bindings, outputs, nil); err == nil {
			t.Fatal("reentrant child construction")
		}
	}
	grandchild, err := child.BeginChildTape(ctx, source, bindings, outputs, []ResidentLeaseEdge{{Source: child, Output: 3, Input: 1}})
	if err != nil {
		t.Fatal(err)
	}
	d.onRun = nil
	if root.Close() == nil || child.Close() == nil || s.terminal != nil {
		t.Fatal("blocked close poisons valid tree")
	}
	if grandchild.Close() != nil || child.Close() != nil || root.Close() != nil || s.Close() != nil {
		t.Fatal("post-reentry close")
	}
}
func TestTrainingResidentTapeChildCrossOwnerAlias(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	root, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	// Child pins the borrowed constant itself. Distinct owner IDs can still
	// represent the same native storage; complete bound-buffer preflight rejects.
	source := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{99, 99}}}, Outputs: []int{0}}
	child, err := root.BeginChildTape(ctx, source, nil, []int{0}, []ResidentLeaseEdge{{Source: root, Output: 2, Input: 0}})
	if err != nil {
		t.Fatal(err)
	}
	consumer, bindings, outputs := multiConsumerFixture()
	before := d.constructions
	if _, err := root.RunSourcesTape(ctx, consumer, bindings, nil, outputs, []ResidentLeaseEdge{{Source: root, Output: 2, Input: 1}, {Source: child, Output: 0, Input: 2}}, true, 1); err == nil || d.constructions != before || s.terminal == nil {
		t.Fatal("crossowner native alias")
	}
	if child.Close() == nil || root.Close() == nil {
		t.Fatal("alias owner released")
	}
}
func TestTrainingResidentTapeChildCompact(t *testing.T) {
	ctx := context.Background()
	snapshot := residentFixture()
	snapshot.Parameters = map[string]ResidentParameter{}
	for name, n := range map[string]int{"a": 8, "b": 2} {
		snapshot.Parameters[name] = ResidentParameter{Parameter: make([]float32, n), M: make([]float32, n), V: make([]float32, n), EMA: make([]float32, n)}
	}
	d := &leaseCompactHostDriver{}
	s, err := NewResidentState(ctx, d, snapshot, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	d.state = s
	producer := Tape{Tensors: []TensorSpec{{Shape: []int{6}, Values: []float32{1, 2, 3, 4, 5, 6}}}, Outputs: []int{0}}
	root, err := s.BeginZeroPrunedTapeLease(ctx, producer, nil, []int{0}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	source := Tape{Tensors: []TensorSpec{{Shape: []int{6}, Values: make([]float32, 6)}, {Shape: []int{6}, Values: make([]float32, 6)}, {Shape: []int{6}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
	child, err := root.BeginChildTape(ctx, source, nil, []int{2}, []ResidentLeaseEdge{{Source: root, Output: 0, Input: 0}})
	if err != nil {
		t.Fatal(err)
	}
	consumer, bindings, compact := compactBindingFixture()
	if _, err := child.RunSourcesTape(ctx, consumer, bindings, compact, nil, []ResidentLeaseEdge{{Source: child, Output: 2, Input: 1}}, true, .5); err != nil {
		t.Fatal(err)
	}
	got := s.entries[s.index["a"]].gradient.(*residentHostBuffer).values
	for i, want := range []float64{1.5, 2, 0, 0, 3, 4, 0, 0} {
		if math.Abs(float64(got[i])-want) > 2e-6 {
			t.Fatal("compact child roots", got)
		}
	}
	if d.compactCalls != 1 || d.denseCalls != 1 {
		t.Fatal("single accumulation")
	}
	if child.Close() != nil || root.Close() != nil || s.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingResidentTapeChildNilAndDeadEdges(t *testing.T) {
	ctx := context.Background()
	if _, err := (*ResidentTapeLease)(nil).BeginChildTape(ctx, Tape{}, nil, nil, nil); err == nil {
		t.Fatal("nil begin")
	}
	if _, err := (*ResidentTapeLease)(nil).RunSourcesTape(ctx, Tape{}, nil, nil, nil, nil, false, 1); err == nil {
		t.Fatal("nil run")
	}
	s, _, root, child := beginChildFixture(t, 1024)
	source := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: make([]float32, 2)}, {Shape: []int{1}, Values: []float32{3}}}, Outputs: []int{1}}
	sibling, err := root.BeginChildTape(ctx, source, nil, []int{1}, []ResidentLeaseEdge{{Source: child, Output: 3, Input: 0}})
	if err != nil {
		t.Fatal(err)
	}
	// The child edge was fully admitted but pruned away. It does not create an
	// artificial live dependency (parent root still cannot close prematurely).
	if child.Close() != nil || sibling.Close() != nil || root.Close() != nil || s.Close() != nil {
		t.Fatal("dead edge pins unnecessary owner")
	}
}

func TestTrainingResidentTapeChildConsumerAdmission(t *testing.T) {
	for _, kind := range []string{"nilcontext", "cancel", "unboundaccum", "foreign", "copy", "edgecount"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			s, d, root, child := beginChildFixture(t, 1024)
			tape, b, outputs := multiConsumerFixture()
			edges := []ResidentLeaseEdge{{Source: child, Output: 3, Input: 1}}
			switch kind {
			case "nilcontext":
				ctx = nil
			case "cancel":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "unboundaccum":
				b = nil
			case "foreign":
				_, _, other, _ := beginChildFixture(t, 1024)
				edges[0].Source = other
			case "copy":
				clone := *child
				edges[0].Source = &clone
			case "edgecount":
				edges = make([]ResidentLeaseEdge, 4097)
			}
			before := d.constructions
			if _, err := root.RunSourcesTape(ctx, tape, b, nil, outputs, edges, true, 1); err == nil || d.constructions != before {
				t.Fatal("consumer admission")
			}
			if kind == "nilcontext" {
				if child.Close() != nil || root.Close() != nil || s.Close() != nil {
					t.Fatal("caller error close")
				}
			} else if s.terminal == nil || child.Close() == nil || root.Close() == nil {
				t.Fatal("live tree not terminal")
			}
		})
	}
}
func TestTrainingResidentTapeChildFrozenGraph(t *testing.T) {
	ctx := context.Background()
	s, _, root, _ := beginChildFixture(t, 1024)
	source, outputs := childFixture()
	_, bindings, _ := leaseConsumerFixture()
	edges := []ResidentLeaseEdge{{Source: root, Output: 2, Input: 1}}
	child, err := root.BeginChildTape(ctx, source, bindings, outputs, edges)
	if err != nil {
		t.Fatal(err)
	}
	source.Tensors[1].Values[0] = 999
	source.Tensors[3].Shape[0] = 1
	source.Commands[0].Inputs[0] = 2
	source.Outputs[0] = 0
	outputs[0] = 0
	edges[0].Input = 2
	delete(bindings, "a")
	values, err := child.DownloadOutputs(ctx, []int{3})
	if err != nil || len(values) != 1 || len(values[0]) != 2 || math.Abs(float64(values[0][0])-1.9) > 2e-6 {
		t.Fatal("child graph borrowed caller metadata", err)
	}
	// Close the fixture's first child as well as this independently retained one.
	for owner := range root.live {
		if owner != root && owner != child {
			if owner.Close() != nil {
				t.Fatal("fixture child close")
			}
		}
	}
	if child.Close() != nil || root.Close() != nil || s.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingResidentTapeChildBeginBoundaries(t *testing.T) {
	for _, kind := range []string{"nilcontext", "extent", "nooutputs", "toooutputs"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			s, d, root, _ := beginChildFixture(t, 1024)
			source, outputs := childFixture()
			_, bindings, _ := leaseConsumerFixture()
			edges := []ResidentLeaseEdge{{Source: root, Output: 2, Input: 1}}
			switch kind {
			case "nilcontext":
				ctx = nil
			case "extent":
				bindings = map[string]ResidentTapeBinding{"b": {Parameter: 0, Gradient: 0}}
			case "nooutputs":
				outputs = nil
			case "toooutputs":
				outputs = make([]int, 4097)
			}
			before := d.constructions
			if _, err := root.BeginChildTape(ctx, source, bindings, outputs, edges); err == nil || d.constructions != before {
				t.Fatal("begin admission")
			}
			if kind != "nilcontext" && s.terminal == nil {
				t.Fatal("begin failure not retained")
			}
		})
	}
}
