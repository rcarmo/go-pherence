package training

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

type leaseHostDriver struct {
	residentHostTapeDriver
	validations, constructions, executions, downloads, accumulations int
	owners                                                           []*leaseHostTape
	closeOrder                                                       []int
	preflightFailure                                                 bool
	onRun                                                            func()
}
type leaseHostTape struct {
	*residentHostBoundTape
	driver *leaseHostDriver
	number int
}

func (d *leaseHostDriver) ValidateHandoffTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) error {
	d.validations++
	if d.preflightFailure {
		return errors.New("lease preflight")
	}
	// Reuse pure mock binding admission, without pair-specific construction count.
	helper := &handoffHostDriver{}
	return helper.ValidateHandoffTape(ctx, t, b, maxBytes)
}
func (d *leaseHostDriver) NewBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	d.constructions++
	base, err := d.residentHostTapeDriver.NewBoundTape(ctx, t, b, maxBytes)
	if base == nil {
		return nil, err
	}
	tape := &leaseHostTape{base.(*residentHostBoundTape), d, d.constructions}
	d.owners = append(d.owners, tape)
	return tape, err
}
func (d *leaseHostDriver) NewReuseBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	return d.NewBoundTape(ctx, t, b, maxBytes)
}
func (d *leaseHostDriver) NewZeroBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	return d.NewBoundTape(ctx, t, b, maxBytes)
}
func (d *leaseHostDriver) Accumulate(ctx context.Context, dst, previous, gradient ResidentBuffer, n int, scale float32) error {
	d.accumulations++
	return d.residentHostDriver.Accumulate(ctx, dst, previous, gradient, n, scale)
}
func (tape *leaseHostTape) RunDevice(ctx context.Context) error {
	tape.driver.executions++
	if tape.driver.onRun != nil {
		tape.driver.onRun()
	}
	if tape.driver.tapeFailure == "run" {
		return errors.New("run")
	}
	for _, c := range tape.tape.Commands {
		out := tape.buffers[c.Output].values
		a := tape.buffers[c.Inputs[0]].values
		switch {
		case c.Op == "vector" && c.Mode == 0:
			b := tape.buffers[c.Inputs[1]].values
			for i := range out {
				out[i] = a[i] + b[i]*c.Scale
			}
		case c.Op == "vector" && c.Mode == 1:
			copy(out, a[c.Offset:c.Offset+len(out)])
		case c.Op == "reshape":
			tape.buffers[c.Output] = tape.buffers[c.Inputs[0]]
		default:
			return errors.New("lease interpreter unsupported")
		}
	}
	return nil
}
func (tape *leaseHostTape) DownloadOutputs(ctx context.Context, ids []int) ([][]float32, error) {
	tape.driver.downloads++
	return tape.residentHostBoundTape.DownloadOutputs(ctx, ids)
}
func (tape *leaseHostTape) Close() error {
	err := tape.residentHostBoundTape.Close()
	if err == nil {
		tape.driver.closeOrder = append(tape.driver.closeOrder, tape.number)
	}
	return err
}
func requireLeaseResident(t *testing.T) (*ResidentState, *leaseHostDriver) {
	t.Helper()
	d := &leaseHostDriver{}
	s, err := NewResidentState(context.Background(), d, residentFixture(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	d.state = s
	return s, d
}
func leaseProducerFixture() (Tape, map[string]ResidentTapeBinding) {
	return residentBindingFixture()
}
func leaseConsumerFixture() (Tape, map[string]ResidentTapeBinding, []TapeHandoffEdge) {
	// Dead leading source and nonidentity pruning remaps exercise edge ID mapping.
	t := Tape{Tensors: []TensorSpec{{Shape: []int{1}, Values: []float32{7}}, {Shape: []int{2}, Values: []float32{99, 99}}, {Shape: []int{2}, Values: []float32{88, 88}}, {Shape: []int{2}}, {Shape: []int{2}, Values: []float32{.4, .2}}}, Commands: []Command{{Op: "vector", Inputs: []int{1, 2}, Output: 3, Scale: 2}}, Outputs: []int{3, 4}}
	return t, map[string]ResidentTapeBinding{"a": {Parameter: 2, Gradient: 4}}, []TapeHandoffEdge{{Output: 2, Input: 1}}
}
func assertLeaseBlocked(t *testing.T, s *ResidentState) {
	t.Helper()
	ctx := context.Background()
	events := 0
	for _, count := range s.driver.(*leaseHostDriver).events {
		events += count
	}
	step := s.StepCount()
	if s.Close() == nil {
		t.Fatal("lease state close")
	}
	if _, err := s.Snapshot(ctx); err == nil {
		t.Fatal("lease snapshot")
	}
	if _, err := s.Update(ctx, 1); err == nil {
		t.Fatal("lease update")
	}
	if s.Accumulate(ctx, map[string][]float32{"a": {1, 2}}, 1) == nil {
		t.Fatal("lease public accumulate")
	}
	tape, bindings := residentBindingFixture()
	if _, err := s.RunTape(ctx, tape, bindings, []int{2}, true, 1, 1024); err == nil {
		t.Fatal("lease public tape")
	}
	if _, err := s.RunUnboundTape(ctx, tape, []int{2}, 1024); err == nil {
		t.Fatal("lease public unbound tape")
	}
	if _, err := s.BeginPrunedTapeLease(ctx, tape, bindings, []int{2}, 1024); err == nil {
		t.Fatal("second lease")
	}
	after := 0
	for _, count := range s.driver.(*leaseHostDriver).events {
		after += count
	}
	if after != events || s.StepCount() != step {
		t.Fatal("blocked state changed")
	}
}
func TestTrainingResidentTapeLeaseMultipleConsumers(t *testing.T) {
	for _, zero := range []bool{false, true} {
		t.Run(map[bool]string{false: "pruned", true: "zero"}[zero], func(t *testing.T) {
			ctx := context.Background()
			s, d := requireLeaseResident(t)
			producer, bindings := leaseProducerFixture()
			var l *ResidentTapeLease
			var err error
			if zero {
				l, err = s.BeginZeroPrunedTapeLease(ctx, producer, bindings, []int{2}, 1024)
			} else {
				l, err = s.BeginPrunedTapeLease(ctx, producer, bindings, []int{2}, 1024)
			}
			if err != nil {
				t.Fatal(err)
			}
			if d.executions != 1 || d.downloads != 0 || s.lease != l || s.borrowed {
				t.Fatal("producer lifetime")
			}
			assertLeaseBlocked(t, s)
			// Consumer callbacks also cannot re-enter the authorised lease while an
			// invocation is borrowing. Never temporarily clear the long-lived lease.
			d.onRun = func() {
				assertLeaseBlocked(t, s)
				if l.Close() == nil {
					t.Fatal("reentrant lease close")
				}
				if _, err := l.DownloadOutputs(ctx, []int{2}); err == nil {
					t.Fatal("reentrant download")
				}
			}
			producer.Tensors[0].Values[0] = 99
			producer.Outputs[0] = 0
			for row := 0; row < 3; row++ {
				tape, b, edges := leaseConsumerFixture()
				values, err := l.RunTape(ctx, tape, b, []int{3}, edges, true, .5)
				if err != nil {
					t.Fatal(err)
				}
				for i, want := range []float64{1.9, -.1} {
					if math.Abs(float64(values[0][i])-want) > 2e-6 {
						t.Fatal("consumer current weights/device root", values)
					}
				}
				// The edge placeholder's stale 99s never enter execution; full source
				// selected root is rebound after nonidentity pruning.
				if d.owners[row+1].buffers[0] != d.owners[0].buffers[l.outputs[2]] || !d.owners[row+1].closed || d.owners[0].closed {
					t.Fatal("pruned edge borrow lifetime")
				}
			}
			d.onRun = nil
			values, err := l.DownloadOutputs(ctx, []int{2})
			if err != nil || len(values) != 1 || math.Abs(float64(values[0][0])-1.3) > 2e-6 {
				t.Fatal("producer selected download", err)
			}
			if d.executions != 4 || d.accumulations != 3 || d.downloads != 4 {
				t.Fatal("accounting", d.executions, d.accumulations, d.downloads)
			}
			entry := s.entries[s.index["a"]]
			for i, want := range []float64{.6, .3} {
				if math.Abs(float64(entry.gradient.(*residentHostBuffer).values[i])-want) > 2e-6 {
					t.Fatal("single-scaled accumulation")
				}
			}
			assertLeaseBlocked(t, s)
			if err = l.Close(); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(d.closeOrder, []int{2, 3, 4, 1}) || s.lease != nil || !d.owners[0].closed {
				t.Fatal("cleanup order", d.closeOrder)
			}
			if l.Close() != nil {
				t.Fatal("idempotent close")
			}
			if _, err := l.DownloadOutputs(ctx, []int{2}); err == nil {
				t.Fatal("expired lease")
			}
			if s.Close() != nil {
				t.Fatal("state close")
			}
		})
	}
}
func TestTrainingResidentTapeLeaseUnboundConsumer(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	_, tape, edges := handoffFixture()
	if _, err = l.RunTape(ctx, tape, nil, []int{2}, edges, false, 1); err != nil {
		t.Fatal(err)
	}
	if s.dirty || d.accumulations != 0 {
		t.Fatal("forward implicit gradients")
	}
	if err = l.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err = s.Snapshot(ctx); err != nil {
		t.Fatal("completed lease snapshot", err)
	}
	if s.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingResidentTapeLeaseAdmission(t *testing.T) {
	for _, kind := range []string{"context", "budget", "source", "binding", "unknown", "extent", "output", "duplicate", "preflight", "capability"} {
		t.Run(kind, func(t *testing.T) {
			s, d := requireLeaseResident(t)
			p, b := leaseProducerFixture()
			ctx := context.Background()
			maxBytes := int64(1024)
			outputs := []int{2}
			switch kind {
			case "context":
				ctx = nil
			case "budget":
				maxBytes = 0
			case "source":
				p.Tensors = append(p.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "binding":
				b["a"] = ResidentTapeBinding{Parameter: 2, Gradient: 0}
			case "unknown":
				b["missing"] = b["a"]
				delete(b, "a")
			case "extent":
				p.Tensors[0] = TensorSpec{Shape: []int{1}, Values: []float32{1}}
			case "output":
				outputs = []int{0}
			case "duplicate":
				outputs = []int{2, 2}
			case "preflight":
				d.preflightFailure = true
			case "capability":
				s.driver = &residentHostTapeDriver{}
			}
			l, err := s.BeginPrunedTapeLease(ctx, p, b, outputs, maxBytes)
			if err == nil || l != nil || s.lease != nil || s.terminal != nil || d.constructions != 0 {
				t.Fatal("producer allocated before admission", err)
			}
		})
	}
}
func TestTrainingResidentTapeLeaseConsumerAdmissionRetention(t *testing.T) {
	for _, kind := range []string{"source", "budget", "edge", "target", "edge-extent", "duplicate", "conflict", "root", "binding", "unbound-accum", "scale", "output", "preflight", "late-accum-preflight"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			s, d := requireLeaseResident(t)
			p, b := leaseProducerFixture()
			limit := int64(1024)
			if kind == "budget" {
				limit = 24
			}
			l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, limit)
			if err != nil {
				t.Fatal(err)
			}
			tape, bindings, edges := leaseConsumerFixture()
			outputs := []int{3}
			accumulate := true
			scale := float32(1)
			switch kind {
			case "source":
				tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "edge":
				edges = []TapeHandoffEdge{{Output: -1, Input: 1}}
			case "target":
				edges[0].Input = 3
			case "edge-extent":
				edges[0].Input = 0
			case "duplicate":
				edges = append(edges, edges[0])
			case "conflict":
				edges[0].Input = 2
			case "root":
				edges[0].Output = 0
			case "binding":
				bindings["bad"] = bindings["a"]
				delete(bindings, "a")
			case "unbound-accum":
				bindings = nil
			case "scale":
				scale = float32(math.NaN())
			case "output":
				outputs = []int{0}
			case "preflight":
				d.preflightFailure = true
			case "late-accum-preflight":
				d.tapeFailure = "admission"
			}
			_, err = l.RunTape(ctx, tape, bindings, outputs, edges, accumulate, scale)
			if err == nil || s.terminal == nil || s.lease != l || d.owners[0].closed || l.Close() == nil {
				t.Fatal("late admission released producer", err)
			}
			if kind != "late-accum-preflight" && d.constructions != 1 {
				t.Fatal("consumer allocation before admission")
			}
			if kind == "late-accum-preflight" && (len(s.retained) != 1 || d.owners[1].closed) {
				t.Fatal("late consumer not retained")
			}
			assertLeaseBlocked(t, s)
		})
	}
}
func TestTrainingResidentTapeLeaseLateRowFailure(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	tape, bindings, edges := leaseConsumerFixture()
	if _, err = l.RunTape(ctx, tape, bindings, []int{3}, edges, true, .5); err != nil {
		t.Fatal(err)
	}
	gradient := append([]float32(nil), s.entries[s.index["a"]].gradient.(*residentHostBuffer).values...)
	d.fail = "accumulate"
	d.at = 2
	d.sentinel = errors.New("late accumulation")
	if _, err = l.RunTape(ctx, tape, bindings, []int{3}, edges, true, .5); err == nil {
		t.Fatal("later-row failure missing")
	}
	if s.terminal == nil || len(s.retained) != 1 || s.StepCount() != 0 || d.owners[0].closed || d.owners[2].closed {
		t.Fatal("uncertain owners not retained")
	}
	if !reflect.DeepEqual(gradient, s.entries[s.index["a"]].gradient.(*residentHostBuffer).values) {
		t.Fatal("previous row rolled back/swapped")
	}
	executions := d.executions
	if _, err = l.RunTape(ctx, tape, bindings, []int{3}, edges, true, .5); err == nil || d.executions != executions {
		t.Fatal("retry")
	}
	if _, err = l.DownloadOutputs(ctx, []int{2}); err == nil {
		t.Fatal("terminal download")
	}
	if l.Close() == nil {
		t.Fatal("terminal producer freed")
	}
	assertLeaseBlocked(t, s)
}
func TestTrainingResidentTapeLeaseProducerFailures(t *testing.T) {
	for _, kind := range []string{"construction", "nil", "run", "close", "download", "count", "extent", "nonfinite"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			s, d := requireLeaseResident(t)
			p, b := leaseProducerFixture()
			if kind == "construction" || kind == "nil" || kind == "run" {
				d.tapeFailure = kind
			}
			l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
			if kind == "construction" || kind == "nil" || kind == "run" {
				if l == nil || err == nil || s.terminal == nil || s.lease != l || l.Close() == nil {
					t.Fatal("producer failure retention", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			d.tapeFailure = kind
			if kind == "close" {
				err = l.Close()
			} else {
				_, err = l.DownloadOutputs(ctx, []int{2})
			}
			if err == nil || s.terminal == nil || d.owners[0].closed || l.Close() == nil {
				t.Fatal("producer output/close failure", err)
			}
		})
	}
}
func TestTrainingResidentTapeLeaseForeignAndReentry(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	other, _ := requireLeaseResident(t)
	copyLease := *l
	if l.admit(ctx, other) == nil || copyLease.admit(ctx, s) == nil {
		t.Fatal("foreign lease")
	}
	tape, bindings, edges := leaseConsumerFixture()
	if _, err = copyLease.RunTape(ctx, tape, bindings, []int{3}, edges, true, 1); err == nil || s.terminal != nil {
		t.Fatal("copied lease authority")
	}
	ctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err = l.RunTape(ctx, tape, bindings, []int{3}, edges, true, 1); err == nil || s.terminal == nil || d.executions != 1 {
		t.Fatal("cancel before authorised consumer")
	}
	if l.Close() == nil || s.Close() == nil {
		t.Fatal("cancelled live producer released")
	}
	if (*ResidentTapeLease)(nil).Close() != nil {
		t.Fatal("nil close")
	}
}

func TestTrainingResidentTapeLeaseCombinedBudget(t *testing.T) {
	for _, limit := range []int64{59, 60} {
		t.Run(map[int64]string{59: "reject", 60: "exact"}[limit], func(t *testing.T) {
			ctx := context.Background()
			s, d := requireLeaseResident(t)
			p, b := leaseProducerFixture()
			l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, limit)
			if err != nil {
				t.Fatal(err)
			}
			consumer, bindings, edges := leaseConsumerFixture() // Full source 36B, producer 24B.
			_, err = l.RunTape(ctx, consumer, bindings, []int{3}, edges, false, 1)
			if limit == 59 {
				if err == nil || d.constructions != 1 || s.terminal == nil {
					t.Fatal("combined late ceiling", err)
				}
				return
			}
			if err != nil || d.constructions != 2 {
				t.Fatal("exact combined ceiling", err)
			}
			if l.Close() != nil || s.Close() != nil {
				t.Fatal("close")
			}
		})
	}
}
func TestTrainingResidentTapeLeaseConsumerFailures(t *testing.T) {
	for _, kind := range []string{"construction", "nil", "run", "output", "download", "count", "extent", "nonfinite", "close"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			s, d := requireLeaseResident(t)
			p, b := leaseProducerFixture()
			l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
			if err != nil {
				t.Fatal(err)
			}
			consumer, bindings, edges := leaseConsumerFixture()
			d.tapeFailure = kind
			_, err = l.RunTape(ctx, consumer, bindings, []int{3}, edges, true, 1)
			if err == nil || s.terminal == nil || s.lease != l || d.owners[0].closed || l.Close() == nil {
				t.Fatal("consumer failure retention", kind, err)
			}
			if kind == "output" {
				// Output borrowing fails before any consumer allocation.
				if d.constructions != 1 || len(s.retained) != 0 {
					t.Fatal("producer output preflight allocated consumer")
				}
			} else if kind != "nil" && (len(s.retained) != 1 || d.owners[1].closed) {
				t.Fatal("failed consumer freed")
			}
			assertLeaseBlocked(t, s)
		})
	}
}
func TestTrainingResidentTapeLeaseExternalRetention(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	first := errors.New("training boundary failed")
	if l.RetainFailure(nil) != nil || l.RetainFailure(first) != first || s.terminal != first || l.Close() == nil || d.owners[0].closed {
		t.Fatal("external failure")
	}
	l.RetainFailure(errors.New("later"))
	if s.terminal != first {
		t.Fatal("first terminal lost")
	}
	assertLeaseBlocked(t, s)
}

type leaseCompactHostDriver struct{ compactHostDriver }

func (d *leaseCompactHostDriver) ValidateHandoffTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) error {
	helper := &handoffHostDriver{}
	return helper.ValidateHandoffTape(ctx, t, b, maxBytes)
}
func TestTrainingResidentTapeLeaseCompactRoots(t *testing.T) {
	for _, failure := range []string{"", "compact-admission", "compact-execution"} {
		t.Run(map[string]string{"": "pass", "compact-admission": "preflight", "compact-execution": "execution"}[failure], func(t *testing.T) {
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
			l, err := s.BeginZeroPrunedTapeLease(ctx, producer, nil, []int{0}, 1024)
			if err != nil {
				t.Fatal(err)
			}
			consumer, b, compact := compactBindingFixture()
			consumer.Tensors[1].Values = []float32{99, 99, 99, 99, 99, 99}
			edges := []TapeHandoffEdge{{Output: 0, Input: 1}}
			d.compactFailure = failure
			_, err = l.RunCompactTape(ctx, consumer, b, compact, nil, edges, true, .5)
			if failure != "" {
				if err == nil || s.terminal == nil || len(s.retained) != 1 || l.Close() == nil {
					t.Fatal("compact failure retention", err)
				}
				return
			}
			if err != nil || d.compactCalls != 1 || d.denseCalls != 1 || d.indexReads != 1 {
				t.Fatal("compact roots/accounting", err, d.indexReads)
			}
			got := s.entries[s.index["a"]].gradient.(*residentHostBuffer).values
			for i, want := range []float64{1.5, 2, 0, 0, 3, 4, 0, 0} {
				if math.Abs(float64(got[i])-want) > 2e-6 {
					t.Fatal("compact source edge", got)
				}
			}
			if l.Close() != nil || s.Close() != nil {
				t.Fatal("close")
			}
		})
	}
}

func TestTrainingResidentTapeLeaseRowSlice(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	producer := Tape{Tensors: []TensorSpec{{Shape: []int{6}, Values: []float32{.1, .2, .3, .4, .5, .6}}}, Outputs: []int{0}}
	l, err := s.BeginPrunedTapeLease(ctx, producer, nil, []int{0}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	for row := 0; row < 3; row++ {
		consumer := Tape{Tensors: []TensorSpec{{Shape: []int{6}, Values: make([]float32, 6)}, {Shape: []int{1}, Values: []float32{0}}, {Shape: []int{2}}, {Shape: []int{2}, Values: []float32{0, 0}}, {Shape: []int{2}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Mode: 1, Offset: row * 2}, {Op: "vector", Inputs: []int{2, 3}, Output: 4, Scale: 1}}, Outputs: []int{4}}
		values, err := l.RunTape(ctx, consumer, nil, []int{4}, []TapeHandoffEdge{{Output: 0, Input: 0}}, false, 1)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range values[0] {
			if math.Abs(float64(v)-float64(producer.Tensors[0].Values[row*2+i])) > 2e-6 {
				t.Fatal("device row slice")
			}
		}
	}
	if d.executions != 4 || d.accumulations != 0 || s.dirty || d.owners[0].closed {
		t.Fatal("row accounting")
	}
	if l.Close() != nil || s.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingResidentTapeLeasePrunedDeadEdge(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	consumer, _, edges := leaseConsumerFixture()
	if _, err = l.RunTape(ctx, consumer, nil, []int{4}, edges, false, 1); err != nil {
		t.Fatal(err)
	}
	if len(d.owners[1].tape.Commands) != 0 || len(d.owners[1].tape.Tensors) != 1 || d.owners[0].closed {
		t.Fatal("dead edge pruning")
	}
	if l.Close() != nil || s.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingResidentTapeLeaseDownloadAdmission(t *testing.T) {
	for _, kind := range []string{"unselected", "cancel", "nil"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			s, d := requireLeaseResident(t)
			p, b := leaseProducerFixture()
			l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
			if err != nil {
				t.Fatal(err)
			}
			ids := []int{2}
			switch kind {
			case "unselected":
				ids = []int{0}
			case "cancel":
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			case "nil":
				ctx = nil
			}
			if _, err = l.DownloadOutputs(ctx, ids); err == nil || d.downloads != 0 {
				t.Fatal("download admission")
			}
			if kind == "nil" {
				if l.Close() != nil || s.Close() != nil {
					t.Fatal("nil caller context close")
				}
			} else if s.terminal == nil || l.Close() == nil {
				t.Fatal("late boundary not terminal")
			}
		})
	}
}
func TestTrainingResidentTapeLeaseReshapeAliasAdmission(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	p.Tensors = append(p.Tensors, TensorSpec{Shape: []int{1, 2}})
	p.Commands = append(p.Commands, Command{Op: "reshape", Inputs: []int{2}, Output: 3})
	p.Outputs = append(p.Outputs, 3)
	l, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2, 3}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	consumer, _, _ := leaseConsumerFixture()
	if _, err = l.RunTape(ctx, consumer, nil, []int{3}, []TapeHandoffEdge{{Output: 2, Input: 1}, {Output: 3, Input: 2}}, false, 1); err == nil || d.constructions != 1 || s.terminal == nil {
		t.Fatal("source reshape alias not admitted early")
	}
}
func TestTrainingResidentTapeLeaseHelpers(t *testing.T) {
	if _, err := leaseTapeBytes(Tape{}, 0); err == nil {
		t.Fatal("budget")
	}
	if _, err := leaseTapeBytes(Tape{Tensors: []TensorSpec{{Shape: []int{0}}}}, 1024); err == nil {
		t.Fatal("extent")
	}
	if _, err := (*ResidentTapeLease)(nil).RunTape(context.Background(), Tape{}, nil, nil, nil, false, 1); err == nil {
		t.Fatal("nil run")
	}
	if _, err := (*ResidentTapeLease)(nil).DownloadOutputs(context.Background(), nil); err == nil {
		t.Fatal("nil download")
	}
}

func TestTrainingResidentTapeLeaseF64Accumulation(t *testing.T) {
	const width = 64
	ctx := context.Background()
	snapshot := residentFixture()
	snapshot.Parameters = map[string]ResidentParameter{"a": {Parameter: make([]float32, width), M: make([]float32, width), V: make([]float32, width), EMA: make([]float32, width)}}
	for i := range snapshot.Parameters["a"].Parameter {
		snapshot.Parameters["a"].Parameter[i] = .1
	}
	d := &leaseHostDriver{}
	s, err := NewResidentState(ctx, d, snapshot, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	d.state = s
	input := make([]float32, width)
	state := uint32(27)
	for i := range input {
		state = state*1664525 + 1013904223
		input[i] = (float32(state>>8)/16777216 - .5) * .5
	}
	producer := Tape{Tensors: []TensorSpec{{Shape: []int{width}, Values: input}}, Outputs: []int{0}}
	l, err := s.BeginPrunedTapeLease(ctx, producer, nil, []int{0}, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	var count, sum, squares, maximum float64
	for row := 0; row < 6; row++ {
		scale := float32(.7 + float32(row)*.1)
		consumer := Tape{Tensors: []TensorSpec{{Shape: []int{width}, Values: make([]float32, width)}, {Shape: []int{width}, Values: make([]float32, width)}, {Shape: []int{width}}}, Commands: []Command{{Op: "vector", Inputs: []int{1, 0}, Output: 2, Scale: scale}}, Outputs: []int{2}}
		values, err := l.RunTape(ctx, consumer, map[string]ResidentTapeBinding{"a": {Parameter: 1, Gradient: 2}}, []int{2}, []TapeHandoffEdge{{Output: 0, Input: 0}}, true, .25)
		if err != nil {
			t.Fatal(err)
		}
		gradient := s.entries[s.index["a"]].gradient.(*residentHostBuffer).values
		for i, x := range input {
			want := float64(float32(.1)) + float64(scale)*float64(x)
			var total float64
			for at := 0; at <= row; at++ {
				total += .25 * (float64(float32(.1)) + float64(float32(.7+float32(at)*.1))*float64(x))
			}
			for _, delta := range []float64{float64(values[0][i]) - want, float64(gradient[i]) - total} {
				error := math.Abs(delta)
				if error > 2e-6 {
					t.Fatalf("lease F64 error %.9g > 2e-6", error)
				}
				count++
				sum += error
				squares += error * error
				maximum = math.Max(maximum, error)
			}
		}
	}
	if d.executions != 7 || d.accumulations != 6 || d.downloads != 6 || d.owners[0].closed {
		t.Fatal("F64 resident accounting")
	}
	if l.Close() != nil || s.Close() != nil {
		t.Fatal("close")
	}
	t.Logf("F64 comparisons %.0f mean_abs %.9g RMS %.9g max_abs %.9g unchanged absolute gate 2e-6", count, sum/count, math.Sqrt(squares/count), maximum)
}

func TestTrainingResidentGradientGroupAdmission(t *testing.T) {
	for _, kind := range []string{"unknown", "duplicate", "empty", "dead-nonfinite"} {
		t.Run(kind, func(t *testing.T) {
			s, d := requireLeaseResident(t)
			source, producerBindings := leaseProducerFixture()
			lease, err := s.BeginZeroPrunedTapeLease(context.Background(), source, producerBindings, []int{2}, 1024)
			if err != nil {
				t.Fatal(err)
			}
			tape, bindings, _ := leaseConsumerFixture()
			names := []string{"a"}
			switch kind {
			case "unknown":
				names = []string{"foreign"}
			case "duplicate":
				names = []string{"a", "a"}
			case "empty":
				names = nil
			case "dead-nonfinite":
				tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			}
			before := d.constructions
			if _, err = lease.RunGradientGroupTape(context.Background(), tape, bindings, nil, nil, names, 1); err == nil || d.constructions != before || s.terminal == nil {
				t.Fatal("group admission/failure retention", kind, err)
			}
			if _, err = lease.RunGradientGroupTape(context.Background(), tape, bindings, nil, nil, []string{"a"}, 1); err == nil || d.constructions != before {
				t.Fatal("group retry")
			}
		})
	}
}

func TestTrainingResidentGradientGroupBindsOtherParameters(t *testing.T) {
	snapshot := residentFixture()
	snapshot.Parameters["b"] = ResidentParameter{Parameter: []float32{.5, .25}, M: make([]float32, 2), V: make([]float32, 2), EMA: []float32{.5, .25}}
	d := &leaseHostDriver{}
	s, err := NewResidentState(context.Background(), d, snapshot, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	d.state = s
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{99, 99}}, {Shape: []int{2}, Values: []float32{88, 88}}, {Shape: []int{2}}, {Shape: []int{2}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: .5}, {Op: "vector", Inputs: []int{1, 0}, Output: 3, Scale: .25}}, Outputs: []int{2, 3}}
	bindings := map[string]ResidentTapeBinding{"a": {Parameter: 0, Gradient: 2}, "b": {Parameter: 1, Gradient: 3}}
	if _, err := s.RunGradientGroupTape(context.Background(), tape, bindings, nil, nil, []string{"a"}, 1, 1024); err != nil {
		t.Fatal(err)
	}
	if _, err := s.RunGradientGroupTape(context.Background(), tape, bindings, nil, nil, []string{"b"}, 1, 1024); err != nil {
		t.Fatal(err)
	}
	for name, binding := range bindings {
		entry := s.entries[s.index[name]]
		got := make([]float32, 2)
		if err := entry.gradient.Download(context.Background(), got); err != nil {
			t.Fatal(err)
		}
		a, b := snapshot.Parameters["a"].Parameter, snapshot.Parameters["b"].Parameter
		for i := range got {
			want := a[i] + .5*b[i]
			if binding.Gradient == 3 {
				want = b[i] + .25*a[i]
			}
			if math.Abs(float64(got[i])-float64(want)) > 2e-6 {
				t.Fatal("nonselected parameter binding lost", name, i, got[i], want)
			}
		}
	}
	if d.executions != 2 || d.accumulations != 2 {
		t.Fatal("group count", d.executions, d.accumulations)
	}
	// Dirty gradients cannot publish a snapshot or skip missing-update checks.
	if _, err := s.Snapshot(context.Background()); err == nil {
		t.Fatal("dirty snapshot admitted")
	}
}
