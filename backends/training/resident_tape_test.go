package training

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

type residentHostTapeDriver struct {
	residentHostDriver
	tapeFailure string
	retained    *residentHostBoundTape
	state       *ResidentState
	hook        bool
}
type residentHostBoundTape struct {
	driver  *residentHostTapeDriver
	tape    Tape
	buffers []*residentHostBuffer
	closed  bool
}

func (d *residentHostTapeDriver) NewBoundTape(ctx context.Context, t Tape, bindings map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	if d.tapeFailure == "nil" {
		return nil, nil
	}
	owner := &residentHostBoundTape{driver: d, tape: t, buffers: make([]*residentHostBuffer, len(t.Tensors))}
	d.retained = owner
	for i, s := range t.Tensors {
		if b, ok := bindings[i]; ok {
			owner.buffers[i] = b.(*residentHostBuffer)
		} else {
			n, _ := s.Elements()
			owner.buffers[i] = &residentHostBuffer{values: make([]float32, n), driver: &d.residentHostDriver}
			copy(owner.buffers[i].values, s.Values)
		}
	}
	if d.tapeFailure == "construction" {
		return owner, errors.New("construction")
	}
	return owner, nil
}
func (d *residentHostTapeDriver) ValidateAccumulate(ctx context.Context, dst, a, b ResidentBuffer, n int, scale float32) error {
	if d.tapeFailure == "admission" {
		return errors.New("admission")
	}
	return nil
}
func (t *residentHostBoundTape) RunDevice(ctx context.Context) error {
	if t.driver.hook {
		if t.driver.state.Close() == nil {
			return errors.New("active borrow closed")
		}
		if _, err := t.driver.state.Snapshot(ctx); err == nil {
			return errors.New("active borrow snapshotted")
		}
		if t.driver.state.Accumulate(ctx, map[string][]float32{"a": {1, 2}}, 1) == nil {
			return errors.New("active borrow accumulated")
		}
		if _, err := t.driver.state.Update(ctx, 1); err == nil {
			return errors.New("active borrow updated")
		}
	}
	if t.driver.tapeFailure == "run" {
		return errors.New("run")
	}
	for _, c := range t.tape.Commands {
		if c.Op != "vector" {
			return errors.New("test interpreter unsupported")
		}
		a, b, out := t.buffers[c.Inputs[0]].values, t.buffers[c.Inputs[1]].values, t.buffers[c.Output].values
		for i := range out {
			out[i] = a[i] + b[i]*c.Scale
		}
	}
	return nil
}
func (t *residentHostBoundTape) OutputBuffer(id int) (ResidentBuffer, error) {
	if t.driver.tapeFailure == "output" {
		return nil, errors.New("output")
	}
	return t.buffers[id], nil
}
func (t *residentHostBoundTape) DownloadOutputs(ctx context.Context, ids []int) ([][]float32, error) {
	if t.driver.tapeFailure == "download" {
		return nil, errors.New("download")
	}
	values := make([][]float32, len(ids))
	for i, id := range ids {
		values[i] = append([]float32(nil), t.buffers[id].values...)
	}
	if t.driver.tapeFailure == "count" {
		return nil, nil
	}
	if t.driver.tapeFailure == "extent" {
		values[0] = nil
	}
	if t.driver.tapeFailure == "nonfinite" {
		values[0][0] = float32(math.Inf(1))
	}
	return values, nil
}
func (t *residentHostBoundTape) Close() error {
	if t.driver.tapeFailure == "close" {
		return errors.New("close")
	}
	t.closed = true
	return nil
}
func residentBindingFixture() (Tape, map[string]ResidentTapeBinding) {
	t := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{99, 99}}, {Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
	return t, map[string]ResidentTapeBinding{"a": {Parameter: 0, Gradient: 2}}
}
func requireBoundResident(t *testing.T) (*ResidentState, *residentHostTapeDriver) {
	t.Helper()
	d := &residentHostTapeDriver{}
	s, err := NewResidentState(context.Background(), d, residentFixture(), 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	d.state = s
	return s, d
}
func TestTrainingResidentTapeBinding(t *testing.T) {
	ctx := context.Background()
	s, d := requireBoundResident(t)
	d.hook = true
	tape, bindings := residentBindingFixture()
	values, err := s.RunTape(ctx, tape, bindings, []int{2}, false, 1, 1024)
	if err != nil {
		t.Fatal(err)
	}
	// Table's stale constants are replaced by current resident parameter values.
	for i, want := range []float32{1.3, 1.3} {
		if values[0][i] != want {
			t.Fatal("resident weights not rebound", values)
		}
	}
	if s.dirty || s.entries[0].ready || !d.retained.closed {
		t.Fatal("nonaccumulating forward changed state")
	}
	values, err = s.RunTape(ctx, tape, bindings, nil, true, .5, 1024)
	if err != nil || len(values) != 0 {
		t.Fatal(err)
	}
	if !s.dirty || !s.entries[0].ready {
		t.Fatal("gradient not accumulated")
	}
	g := s.entries[0].gradient.(*residentHostBuffer).values
	for _, x := range g {
		if x != .65 {
			t.Fatal("device gradient accumulation", g)
		}
	}
	if _, err := s.Snapshot(ctx); err == nil {
		t.Fatal("dirty tape snapshot")
	}
	if err := s.Accumulate(ctx, map[string][]float32{"b": {0}}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Update(ctx, 1); err != nil {
		t.Fatal(err)
	}
	snapshot, err := s.Snapshot(ctx)
	if err != nil {
		t.Fatal(err)
	}
	before := snapshot.Parameters["a"].Parameter[0]
	values, err = s.RunTape(ctx, tape, bindings, []int{2}, false, 1, 1024)
	if err != nil || values[0][0] != before+1 {
		t.Fatal("updated resident weights not read", err, values)
	}
	if err = s.Close(); err != nil {
		t.Fatal(err)
	}
}
func TestTrainingResidentTapeAdmission(t *testing.T) {
	tape, bindings := residentBindingFixture()
	if err := ValidateResidentTapeBindings(tape, bindings, 1024); err != nil {
		t.Fatal(err)
	}
	for _, b := range []map[string]ResidentTapeBinding{nil, {"a": {Parameter: -1, Gradient: 2}}, {"a": {Parameter: 2, Gradient: 0}}, {"a": {Parameter: 0, Gradient: 3}}, {"a": {Parameter: 0, Gradient: 0}}, {"a": {Parameter: 0, Gradient: 2}, "b": {Parameter: 0, Gradient: 2}}, {"": {Parameter: 0, Gradient: 2}}} {
		if ValidateResidentTapeBindings(tape, b, 1024) == nil {
			t.Fatal("invalid binding")
		}
	}
	if ValidateResidentTapeBindings(Tape{}, bindings, 1024) == nil {
		t.Fatal("invalid tape")
	}
	if ValidateResidentTapeBindings(tape, bindings, 1) == nil {
		t.Fatal("budget")
	}
	ctx := context.Background()
	s, _ := requireBoundResident(t)
	unknown := map[string]ResidentTapeBinding{"bad": {Parameter: 0, Gradient: 2}}
	if _, err := s.RunTape(ctx, tape, unknown, nil, false, 1, 1024); err == nil {
		t.Fatal("unknown parameter")
	}
	if _, err := s.RunTape(ctx, tape, bindings, []int{0}, false, 1, 1024); err == nil {
		t.Fatal("undeclared output")
	}
	unbound, _ := requireResident(t)
	if _, err := unbound.RunTape(ctx, tape, bindings, nil, false, 1, 1024); err == nil {
		t.Fatal("unsupported driver")
	}
	if _, err := (*ResidentState)(nil).RunTape(ctx, tape, bindings, nil, false, 1, 1024); err == nil {
		t.Fatal("nil resident")
	}
}
func TestTrainingResidentTapeFailureRetention(t *testing.T) {
	ctx := context.Background()
	for _, category := range []string{"construction", "nil", "run", "output", "download", "count", "extent", "nonfinite", "close", "admission", "accumulate"} {
		t.Run(category, func(t *testing.T) {
			s, d := requireBoundResident(t)
			before, _ := s.Snapshot(ctx)
			tape, bindings := residentBindingFixture()
			d.tapeFailure = category
			if category == "accumulate" {
				d.fail, d.at, d.sentinel = "accumulate", 1, errors.New("accumulate")
			}
			if _, err := s.RunTape(ctx, tape, bindings, []int{2}, true, 1, 1024); err == nil {
				t.Fatal("failed tape accepted")
			}
			if s.borrowed {
				t.Fatal("active borrow stuck")
			}
			if category == "admission" {
				if s.terminal != nil || !d.retained.closed || s.dirty {
					t.Fatal("pure preflight mutated resident")
				}
				after, err := s.Snapshot(ctx)
				if err != nil || !reflect.DeepEqual(before, after) {
					t.Fatal("preflight snapshot changed")
				}
				return
			}
			if s.terminal == nil || s.Close() == nil {
				t.Fatal("uncertain resources released")
			}
			if category != "nil" && len(s.retained) != 1 {
				t.Fatal("tape owner lost")
			}
			if _, err := s.RunTape(ctx, tape, bindings, nil, false, 1, 1024); err == nil {
				t.Fatal("failed tape retried")
			}
		})
	}
}
