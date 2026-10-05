package training

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
)

// CPU-only driver intentionally restricts its interpreter to the primitives
// needed by these fixtures. Native paths are never opened by this test.
type compactHostDriver struct {
	residentHostTapeDriver
	validations, compactCalls, denseCalls, indexReads, constructions int
	compactFailure                                                   string
	last                                                             Tape
}
type compactHostTape struct {
	*residentHostBoundTape
	owner *compactHostDriver
}

func (d *compactHostDriver) NewBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	d.constructions++
	d.last = t
	base, err := d.residentHostTapeDriver.NewBoundTape(ctx, t, b, maxBytes)
	if base == nil {
		return nil, err
	}
	return &compactHostTape{base.(*residentHostBoundTape), d}, err
}
func (d *compactHostDriver) NewReuseBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	return d.NewBoundTape(ctx, t, b, maxBytes)
}
func (d *compactHostDriver) NewZeroBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	return d.NewBoundTape(ctx, t, b, maxBytes)
}
func (t *compactHostTape) RunDevice(ctx context.Context) error {
	if t.owner.hook {
		if t.owner.state.Close() == nil {
			return errors.New("active close")
		}
		if _, err := t.owner.state.Snapshot(ctx); err == nil {
			return errors.New("active snapshot")
		}
		if t.owner.state.Accumulate(ctx, map[string][]float32{"a": make([]float32, 8)}, 1) == nil {
			return errors.New("active accumulate")
		}
		if _, err := t.owner.state.Update(ctx, 1); err == nil {
			return errors.New("active update")
		}
	}
	if t.owner.tapeFailure == "run" {
		return errors.New("run")
	}
	for _, c := range t.tape.Commands {
		out := t.buffers[c.Output].values
		if c.Op == "embedding" && c.Mode == 3 {
			source, idx := t.buffers[c.Inputs[0]].values, t.buffers[c.Inputs[1]].values
			for i := range out {
				slot := i / c.Width
				if idx[slot] < 0 {
					out[i] = 0
					continue
				}
				for at := int(idx[c.Rows+slot]); at < int(idx[c.Rows+slot+1]); at++ {
					row := int(idx[2*c.Rows+1+at])
					out[i] += source[row*c.Width+i%c.Width]
				}
			}
		} else if c.Op == "vector" {
			a, b := t.buffers[c.Inputs[0]].values, t.buffers[c.Inputs[1]].values
			for i := range out {
				out[i] = a[i] + c.Scale*b[i]
			}
		}
	}
	return nil
}
func (t *compactHostTape) OutputBuffer(id int) (ResidentBuffer, error) {
	if t.closed {
		return nil, errors.New("expired output")
	}
	if len(t.tape.Tensors[id].Values) > 0 && t.owner.compactFailure == "index-output" {
		return nil, errors.New("index output")
	}
	for _, c := range t.tape.Commands {
		if c.Op == "embedding" && c.Inputs[1] == id {
			t.owner.indexReads++
		}
	}
	return t.residentHostBoundTape.OutputBuffer(id)
}
func (d *compactHostDriver) ValidateCompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index ResidentBuffer, rows, width, vocabulary int, scale float32) error {
	d.validations++
	if d.compactFailure == "compact-admission" {
		return errors.New("compact admission")
	}
	dense, slots, indices, err := CompactEmbeddingAccumulateShape(rows, width, vocabulary, scale)
	if err != nil {
		return err
	}
	for i, x := range []ResidentBuffer{out, previous, compact, index} {
		n := []int{dense, dense, slots, indices}[i]
		if len(x.(*residentHostBuffer).values) != n {
			return errors.New("extent")
		}
	}
	return nil
}
func (d *compactHostDriver) ValidateAccumulate(ctx context.Context, out, previous, source ResidentBuffer, n int, scale float32) error {
	d.validations++
	return d.residentHostTapeDriver.ValidateAccumulate(ctx, out, previous, source, n, scale)
}
func (d *compactHostDriver) CompactEmbeddingAccumulate(ctx context.Context, out, previous, compact, index ResidentBuffer, rows, width, vocabulary int, scale float32) error {
	if d.validations < 2 {
		return errors.New("partial preflight")
	}
	d.compactCalls++
	if d.compactFailure == "compact-execution" {
		return errors.New("compact execution")
	}
	o, p, c, idx := out.(*residentHostBuffer).values, previous.(*residentHostBuffer).values, compact.(*residentHostBuffer).values, index.(*residentHostBuffer).values
	for i := range o {
		token := i / width
		var contribution float32
		for slot := 0; slot < rows; slot++ {
			if idx[slot] == float32(token) {
				contribution = c[slot*width+i%width]
				break
			}
		}
		o[i] = p[i] + scale*contribution
	}
	return nil
}
func (d *compactHostDriver) Accumulate(ctx context.Context, out, previous, source ResidentBuffer, n int, scale float32) error {
	if d.validations < 2 {
		return errors.New("partial preflight")
	}
	d.denseCalls++
	return d.residentHostDriver.Accumulate(ctx, out, previous, source, n, scale)
}

func compactBindingFixture() (Tape, map[string]ResidentTapeBinding, map[string]ResidentCompactEmbeddingBinding) {
	idx, _ := BuildCompactEmbeddingIndex([]float32{2, 0, 2}, 4)
	tape := Tape{Tensors: []TensorSpec{
		{Shape: []int{8}, Values: make([]float32, 8)},
		{Shape: []int{6}, Values: []float32{1, 2, 3, 4, 5, 6}},
		{Shape: []int{10}, Values: idx}, {Shape: []int{6}},
		{Shape: []int{2}, Values: []float32{99, 99}},
		{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}},
	}, Commands: []Command{{Op: "embedding", Inputs: []int{1, 2}, Output: 3, Rows: 3, Width: 2, Parts: 4, Mode: 3}, {Op: "vector", Inputs: []int{4, 5}, Output: 6, Scale: 1}}, Outputs: []int{3, 2, 6}}
	return tape, map[string]ResidentTapeBinding{"a": {Parameter: 0, Gradient: 3}, "b": {Parameter: 4, Gradient: 6}}, map[string]ResidentCompactEmbeddingBinding{"a": {Index: 2, Rows: 3, Width: 2, Vocabulary: 4}}
}
func compactStateFixture(t *testing.T) (*ResidentState, *compactHostDriver) {
	t.Helper()
	snapshot := residentFixture()
	snapshot.Parameters = map[string]ResidentParameter{}
	for name, n := range map[string]int{"a": 8, "b": 2} {
		snapshot.Parameters[name] = ResidentParameter{Parameter: make([]float32, n), M: make([]float32, n), V: make([]float32, n), EMA: make([]float32, n)}
	}
	d := &compactHostDriver{}
	s, err := NewResidentState(context.Background(), d, snapshot, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	d.state = s
	return s, d
}
func TestTrainingResidentCompactBindingsAdmission(t *testing.T) {
	tape, bindings, c := compactBindingFixture()
	if ValidateResidentTapeBindings(tape, bindings, 1<<20) == nil {
		t.Fatal("legacy relaxed")
	}
	if err := ValidateResidentCompactTapeBindings(tape, bindings, c, 1<<20); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"nil", "empty", "name", "index", "notdeclared", "reboundindex", "shape", "extent", "provenance", "producer", "deadsource"} {
		t.Run(kind, func(t *testing.T) {
			tape, b, c := compactBindingFixture()
			x := c["a"]
			switch kind {
			case "nil":
				c = nil
			case "empty":
				c = map[string]ResidentCompactEmbeddingBinding{}
			case "name":
				delete(c, "a")
				c["unknown"] = x
			case "index":
				x.Index = -1
				c["a"] = x
			case "notdeclared":
				tape.Outputs = []int{3, 6}
			case "reboundindex":
				tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{10}, Values: make([]float32, 10)})
				tape.Outputs = append(tape.Outputs, 7)
				b["c"] = ResidentTapeBinding{Parameter: 2, Gradient: 7}
			case "shape":
				x.Rows = 0
				c["a"] = x
			case "extent":
				x.Width = 1
				c["a"] = x
			case "provenance":
				tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{10}, Values: append([]float32(nil), tape.Tensors[2].Values...)})
				tape.Outputs = append(tape.Outputs, 7)
				x.Index = 7
				c["a"] = x
			case "producer":
				tape.Tensors[3].Values = make([]float32, 6)
				tape.Commands = tape.Commands[1:]
			case "deadsource":
				tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			}
			if err := ValidateResidentCompactTapeBindings(tape, b, c, 1<<20); err == nil {
				t.Fatal("invalid compact admitted", kind)
			}
		})
	}
}
func TestTrainingResidentCompactTapeAccumulation(t *testing.T) {
	for _, scale := range []float32{0, .25, 1, -.5} {
		s, d := compactStateFixture(t)
		d.hook = true
		tape, b, c := compactBindingFixture()
		before, _ := json.Marshal(tape)
		for repeat := 0; repeat < 2; repeat++ {
			d.validations = 0
			out, err := s.RunCompactZeroPrunedTape(context.Background(), tape, b, c, []int{6}, true, scale, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			if len(out) != 1 || len(out[0]) != 2 || out[0][0] != 1 || out[0][1] != 2 {
				t.Fatal("boundary", out)
			}
			factor := float32(repeat+1) * scale
			for i, want := range []float32{3, 4, 0, 0, 6, 8, 0, 0} {
				got := s.entries[s.index["a"]].gradient.(*residentHostBuffer).values[i]
				if math.Abs(float64(got-factor*want)) > 2e-6 {
					t.Fatal("compact sum/scale", i, got)
				}
			}
			if !d.retained.closed || s.borrowed || d.indexReads != repeat+1 || d.compactCalls != repeat+1 || d.denseCalls != repeat+1 {
				t.Fatal("lifetime/accounting")
			}
			// Parameter a is unused by its compact VJP, so pruning may omit it. Both
			// compact gradient and constant index must survive as declared outputs.
			hasIndex := false
			for _, id := range d.last.Outputs {
				if len(d.last.Tensors[id].Values) == 10 {
					hasIndex = true
				}
			}
			if !hasIndex {
				t.Fatal("index root pruned")
			}
		}
		after, _ := json.Marshal(tape)
		if string(before) != string(after) {
			t.Fatal("source mutated")
		}
		if _, err := s.Snapshot(context.Background()); err == nil {
			t.Fatal("dirty snapshot")
		}
		if _, err := s.Update(context.Background(), 1e20); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Snapshot(context.Background()); err != nil {
			t.Fatal(err)
		}
		if s.Close() != nil {
			t.Fatal("close")
		}
	}
}
func TestTrainingResidentCompactTapeNoAccumulation(t *testing.T) {
	s, d := compactStateFixture(t)
	tape, b, c := compactBindingFixture()
	if _, err := s.RunCompactZeroPrunedTape(context.Background(), tape, b, c, []int{6}, false, 1, 1<<20); err != nil {
		t.Fatal(err)
	}
	if d.compactCalls != 0 || d.denseCalls != 0 || s.dirty {
		t.Fatal("forward accumulated")
	}
	for _, command := range d.last.Commands {
		if command.Op == "embedding" {
			t.Fatal("unused VJP not pruned")
		}
	}
	if s.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingResidentCompactTapeFailures(t *testing.T) {
	for _, kind := range []string{"compact-admission", "admission", "compact-execution", "accumulate", "index-output", "output", "run", "construction", "nil", "download", "count", "extent", "nonfinite", "close"} {
		t.Run(kind, func(t *testing.T) {
			s, d := compactStateFixture(t)
			tape, b, c := compactBindingFixture()
			d.compactFailure = kind
			d.tapeFailure = kind
			if kind == "accumulate" {
				d.fail, d.at, d.sentinel = "accumulate", 1, errors.New("accumulate")
			}
			_, err := s.RunCompactZeroPrunedTape(context.Background(), tape, b, c, []int{6}, true, 1, 1<<20)
			if err == nil {
				t.Fatal("failure accepted")
			}
			if strings.Contains(kind, "admission") {
				if s.dirty || d.compactCalls != 0 || d.denseCalls != 0 || s.terminal != nil || !d.retained.closed {
					t.Fatal("preflight mutated/retained")
				}
				if s.Close() != nil {
					t.Fatal("pure rejection close")
				}
			} else {
				if s.terminal == nil || s.Close() == nil {
					t.Fatal("terminal owner released")
				}
				calls := d.constructions
				if _, err := s.RunCompactZeroPrunedTape(context.Background(), tape, b, c, []int{6}, true, 1, 1<<20); err == nil || calls != d.constructions {
					t.Fatal("retry")
				}
			}
		})
	}
}
func TestTrainingResidentCompactTapeEarlyAdmission(t *testing.T) {
	for _, kind := range []string{"nilbindings", "unknown", "budget", "scale", "outputs"} {
		s, d := compactStateFixture(t)
		tape, b, c := compactBindingFixture()
		budget := int64(1 << 20)
		scale := float32(1)
		outputs := []int{6}
		switch kind {
		case "nilbindings":
			c = nil
		case "unknown":
			b["unknown"] = b["b"]
			delete(b, "b")
		case "budget":
			budget = 1
		case "scale":
			scale = float32(math.NaN())
		case "outputs":
			outputs = []int{0}
		}
		if _, err := s.RunCompactZeroPrunedTape(context.Background(), tape, b, c, outputs, true, scale, budget); err == nil || d.constructions != 0 {
			t.Fatal("early admission", kind, err)
		}
		if s.Close() != nil {
			t.Fatal("close")
		}
	}
	// Driver capabilities must be admitted before any graph/native work.
	s, d := requireBoundResident(t)
	tape, b, c := compactBindingFixture()
	if _, err := s.RunCompactZeroPrunedTape(context.Background(), tape, b, c, []int{6}, true, 1, 1<<20); err == nil || d.retained != nil {
		t.Fatal("driver capability")
	}
	if s.Close() != nil {
		t.Fatal("close")
	}
}

func TestTrainingResidentCompactEqualExtentRejected(t *testing.T) {
	tape, b, c := compactBindingFixture()
	idx, _ := BuildCompactEmbeddingIndex([]float32{2, 0, 2, 1}, 4)
	tape.Tensors[1] = TensorSpec{Shape: []int{8}, Values: make([]float32, 8)}
	tape.Tensors[2] = TensorSpec{Shape: []int{13}, Values: idx}
	tape.Tensors[3] = TensorSpec{Shape: []int{8}}
	tape.Commands[0].Rows = 4
	c["a"] = ResidentCompactEmbeddingBinding{Index: 2, Rows: 4, Width: 2, Vocabulary: 4}
	if ValidateResidentTapeBindings(tape, b, 1<<20) == nil {
		t.Fatal("same-size compact masqueraded as dense")
	}
	if err := ValidateResidentCompactTapeBindings(tape, b, c, 1<<20); err != nil {
		t.Fatal(err)
	}
	tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{8}})
	tape.Commands = append(tape.Commands, Command{Op: "reshape", Inputs: []int{3}, Output: 7})
	tape.Outputs = append(tape.Outputs, 7)
	b["a"] = ResidentTapeBinding{Parameter: 0, Gradient: 7}
	if ValidateResidentTapeBindings(tape, b, 1<<20) == nil {
		t.Fatal("compact reshape masqueraded as dense")
	}
	if ValidateResidentCompactTapeBindings(tape, b, c, 1<<20) == nil {
		t.Fatal("indirect compact producer admitted")
	}
}

// Expose legacy zero/reuse capabilities but not the compact interface, so the
// new capability gate itself is tested rather than an earlier storage gate.
type denseOnlyZeroDriver struct{ *residentHostTapeDriver }

func (d *denseOnlyZeroDriver) NewReuseBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	return d.NewBoundTape(ctx, t, b, maxBytes)
}
func (d *denseOnlyZeroDriver) NewZeroBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	return d.NewBoundTape(ctx, t, b, maxBytes)
}
func TestTrainingResidentCompactDriverCapability(t *testing.T) {
	s, d := requireBoundResident(t)
	s.driver = &denseOnlyZeroDriver{d}
	tape, b, c := compactBindingFixture()
	_, err := s.RunCompactZeroPrunedTape(context.Background(), tape, b, c, []int{6}, true, 1, 1<<20)
	if err == nil || !strings.Contains(err.Error(), "lacks compact embedding accumulation") || d.retained != nil {
		t.Fatal("compact capability", err)
	}
	if s.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingResidentCompactInventoryExtent(t *testing.T) {
	// Complete binding validation succeeds but resident parameter extent does
	// not: rejection still precedes allocation and state mutation.
	s, d := compactStateFixture(t)
	tape, b, c := compactBindingFixture()
	tape.Tensors[4] = TensorSpec{Shape: []int{1}, Values: []float32{1}}
	tape.Tensors[5] = TensorSpec{Shape: []int{1}, Values: []float32{1}}
	tape.Tensors[6] = TensorSpec{Shape: []int{1}}
	_, err := s.RunCompactZeroPrunedTape(context.Background(), tape, b, c, []int{6}, true, 1, 1<<20)
	if err == nil || !strings.Contains(err.Error(), "extent") || d.constructions != 0 || s.dirty {
		t.Fatal("resident extent", err)
	}
	if s.Close() != nil {
		t.Fatal("close")
	}
}
