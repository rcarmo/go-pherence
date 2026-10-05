package training

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"
)

type handoffHostDriver struct {
	residentHostTapeDriver
	validations, constructions, runs, downloads, outputs int
	failure                                              string
	owners                                               []*handoffHostTape
	closeOrder                                           []int
}
type handoffHostTape struct {
	*residentHostBoundTape
	driver *handoffHostDriver
	number int
}

func (d *handoffHostDriver) ValidateHandoffTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) error {
	d.validations++
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := t.Validate(maxBytes); err != nil {
		return err
	}
	if d.failure == "preflight" && d.validations == 2 {
		return errors.New("consumer preflight")
	}
	if d.failure == "late-preflight" && d.validations == 3 {
		return errors.New("late bindings")
	}
	seen := map[*residentHostBuffer]bool{}
	for id, value := range b {
		if id < 0 || id >= len(t.Tensors) || (t.Tensors[id].Values == nil && !t.Tensors[id].External) {
			return errors.New("bound ID")
		}
		buffer, ok := value.(*residentHostBuffer)
		if !ok || buffer == nil {
			return errors.New("bound owner")
		}
		n, _ := t.Tensors[id].Elements()
		if len(buffer.values) != n || seen[buffer] {
			return errors.New("bound extent/alias")
		}
		seen[buffer] = true
	}
	return nil
}
func (d *handoffHostDriver) Accumulate(ctx context.Context, dst, previous, gradient ResidentBuffer, n int, scale float32) error {
	if d.failure == "external-accumulation" {
		return errors.New("external accumulation uncertain")
	}
	return d.residentHostDriver.Accumulate(ctx, dst, previous, gradient, n, scale)
}
func (d *handoffHostDriver) NewBoundTape(ctx context.Context, t Tape, b map[int]ResidentBuffer, maxBytes int64) (ResidentBoundTape, error) {
	d.constructions++
	if d.validations < 2 {
		return nil, errors.New("construction before full preflight")
	}
	base, err := d.residentHostTapeDriver.NewBoundTape(ctx, t, b, maxBytes)
	if err != nil || base == nil {
		return base, err
	}
	owner := &handoffHostTape{base.(*residentHostBoundTape), d, d.constructions}
	d.owners = append(d.owners, owner)
	if d.failure == "construction" && owner.number == 2 {
		return owner, errors.New("consumer construction")
	}
	if d.failure == "producer-construction" && owner.number == 1 {
		return owner, errors.New("producer construction")
	}
	if d.failure == "nil-consumer" && owner.number == 2 {
		return nil, nil
	}
	if d.failure == "nil-producer" && owner.number == 1 {
		return nil, nil
	}
	return owner, nil
}
func (t *handoffHostTape) RunDevice(ctx context.Context) error {
	t.driver.runs++
	if t.driver.failure == "producer-run" && t.number == 1 || t.driver.failure == "consumer-run" && t.number == 2 {
		return errors.New("run")
	}
	if t.number == 2 && t.driver.owners[0].closed {
		return errors.New("producer expired")
	}
	return t.residentHostBoundTape.RunDevice(ctx)
}
func (t *handoffHostTape) OutputBuffer(id int) (ResidentBuffer, error) {
	t.driver.outputs++
	if t.driver.failure == "producer-output" && t.number == 1 || t.driver.failure == "consumer-output" && t.number == 2 {
		return nil, errors.New("output")
	}
	return t.residentHostBoundTape.OutputBuffer(id)
}
func (t *handoffHostTape) DownloadOutputs(ctx context.Context, ids []int) ([][]float32, error) {
	t.driver.downloads++
	if t.number != 2 {
		return nil, errors.New("handoff downloaded producer")
	}
	if t.driver.failure == "download" {
		return nil, errors.New("download")
	}
	return t.residentHostBoundTape.DownloadOutputs(ctx, ids)
}
func (t *handoffHostTape) Close() error {
	if t.driver.failure == "consumer-close" && t.number == 2 || t.driver.failure == "producer-close" && t.number == 1 {
		return errors.New("close")
	}
	t.driver.closeOrder = append(t.driver.closeOrder, t.number)
	return t.residentHostBoundTape.Close()
}
func handoffFixture() (Tape, Tape, []TapeHandoffEdge) {
	source := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: .5}}, Outputs: []int{2}}
	target := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{99, 99}}, {Shape: []int{2}, Values: []float32{1, 1}}, {Shape: []int{2}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 2}}, Outputs: []int{2}}
	return source, target, []TapeHandoffEdge{{Output: 2, Input: 0}}
}
func TestTrainingTapeHandoffOwnedLifetime(t *testing.T) {
	source, target, edges := handoffFixture()
	d := &handoffHostDriver{}
	p, err := NewResidentTapeHandoff(context.Background(), d, source, target, edges, nil, nil, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if d.constructions != 1 || d.runs != 0 || d.validations != 2 {
		t.Fatal("construction preflight")
	}
	if _, err := p.OutputBuffer(2); err == nil {
		t.Fatal("premature output")
	}
	// Mutating caller-owned metadata/values/edge lists after construction cannot
	// alter either frozen graph or the deferred consumer's input mapping.
	source.Tensors[0].Values[0] = 88
	target.Tensors[1].Values[0] = 88
	source.Commands[0].Inputs[0] = 1
	target.Commands[0].Inputs[0] = 1
	edges[0].Input = 1
	target.Outputs[0] = 0
	if err := p.RunDevice(context.Background()); err != nil {
		t.Fatal(err)
	}
	if d.downloads != 0 || d.runs != 2 || d.owners[0].closed {
		t.Fatal("handoff/lifetime")
	}
	if d.owners[1].buffers[0] != d.owners[0].buffers[2] {
		t.Fatal("device edge not bound")
	}
	out, err := p.OutputBuffer(2)
	if err != nil {
		t.Fatal(err)
	}
	got := out.(*residentHostBuffer).values
	for i, want := range []float32{4.5, 6} {
		if math.Abs(float64(got[i]-want)) > 2e-6 {
			t.Fatal("numeric handoff", got)
		}
	}
	values, err := p.DownloadOutputs(context.Background(), []int{2})
	if err != nil || len(values) != 1 || d.downloads != 1 {
		t.Fatal("selected boundary", err)
	}
	if _, err := p.OutputBuffer(0); err == nil {
		t.Fatal("undeclared output")
	}
	if _, err := p.DownloadOutputs(context.Background(), []int{0}); err == nil {
		t.Fatal("undeclared download")
	}
	if err := p.RunDevice(context.Background()); err == nil {
		t.Fatal("retry")
	}
	if err := p.Close(); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(d.closeOrder, []int{2, 1}) {
		t.Fatal("producer freed before consumer", d.closeOrder)
	}
	if p.Close() != nil {
		t.Fatal("idempotent close")
	}
	if _, err := p.OutputBuffer(2); err == nil {
		t.Fatal("closed output")
	}
	if (*ResidentTapeHandoff)(nil).Close() != nil {
		t.Fatal("nil close")
	}
}
func TestTrainingTapeHandoffAdmission(t *testing.T) {
	for _, kind := range []string{"context", "driver", "budget", "source", "target", "edges", "id", "constant", "duplicate", "producer-duplicate", "producer-reshape-alias", "extent", "binding", "preflight"} {
		t.Run(kind, func(t *testing.T) {
			s, c, e := handoffFixture()
			d := &handoffHostDriver{}
			ctx := context.Background()
			budget := int64(1024)
			var driver ResidentHandoffTapeDriver = d
			var bindings map[int]ResidentBuffer
			switch kind {
			case "context":
				ctx = nil
			case "driver":
				driver = nil
			case "budget":
				budget = 0
			case "source":
				s.Tensors = append(s.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "target":
				c.Tensors = append(c.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "edges":
				e = nil
			case "id":
				e[0].Output = 0
			case "constant":
				e[0].Input = 2
			case "duplicate":
				e = append(e, e[0])
			case "producer-duplicate":
				e = append(e, TapeHandoffEdge{Output: 2, Input: 1})
			case "producer-reshape-alias":
				s.Tensors = append(s.Tensors, TensorSpec{Shape: []int{1, 2}})
				s.Commands = append(s.Commands, Command{Op: "reshape", Inputs: []int{2}, Output: 3})
				s.Outputs = append(s.Outputs, 3)
				e = append(e, TapeHandoffEdge{Output: 3, Input: 1})
			case "extent":
				c.Tensors = append(c.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{1}})
				e[0].Input = 3
			case "binding":
				bindings = map[int]ResidentBuffer{0: &residentHostBuffer{values: make([]float32, 2)}}
			case "preflight":
				d.failure = "preflight"
			}
			if p, err := NewResidentTapeHandoff(ctx, driver, s, c, e, nil, bindings, budget); err == nil || p != nil || d.constructions != 0 {
				t.Fatal("late graph admission", kind, err)
			}
		})
	}
}
func TestTrainingTapeHandoffFailureRetention(t *testing.T) {
	for _, kind := range []string{"producer-construction", "nil-producer", "producer-run", "producer-output", "late-preflight", "construction", "nil-consumer", "consumer-run", "consumer-output", "download", "consumer-close", "producer-close"} {
		t.Run(kind, func(t *testing.T) {
			s, c, e := handoffFixture()
			d := &handoffHostDriver{failure: kind}
			p, err := NewResidentTapeHandoff(context.Background(), d, s, c, e, nil, nil, 1024)
			if kind == "producer-construction" || kind == "nil-producer" {
				if err == nil || p == nil || p.Close() == nil {
					t.Fatal("partial constructor owner")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			err = p.RunDevice(context.Background())
			switch kind {
			case "consumer-output":
				if err != nil {
					t.Fatal(err)
				}
				_, err = p.OutputBuffer(2)
			case "download":
				if err != nil {
					t.Fatal(err)
				}
				_, err = p.DownloadOutputs(context.Background(), []int{2})
			case "consumer-close", "producer-close":
				if err != nil {
					t.Fatal(err)
				}
				err = p.Close()
			}
			if err == nil || p.terminal == nil || p.Close() == nil {
				t.Fatal("failure not retained", kind)
			}
			if kind == "consumer-close" && d.owners[0].closed {
				t.Fatal("producer released under failed consumer")
			}
			if kind == "producer-close" && !d.owners[1].closed {
				t.Fatal("consumer close order")
			}
			outputs, downloads := d.outputs, d.downloads
			if _, err := p.OutputBuffer(2); err == nil {
				t.Fatal("terminal output access")
			}
			if _, err := p.DownloadOutputs(context.Background(), []int{2}); err == nil || d.outputs != outputs || d.downloads != downloads {
				t.Fatal("terminal access reached driver")
			}
			runs := d.runs
			if p.RunDevice(context.Background()) == nil || d.runs != runs {
				t.Fatal("uncertain retry")
			}
		})
	}
}
func TestTrainingTapeHandoffCancelledBeforeRun(t *testing.T) {
	s, c, e := handoffFixture()
	d := &handoffHostDriver{}
	p, err := NewResidentTapeHandoff(context.Background(), d, s, c, e, nil, nil, 1024)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if p.RunDevice(ctx) == nil || d.runs != 0 || p.terminal != nil {
		t.Fatal("cancel before execution")
	}
	if p.Close() != nil {
		t.Fatal("close")
	}
}

func TestTrainingTapeHandoffCombinedBudget(t *testing.T) {
	source, target, edges := handoffFixture()
	d := &handoffHostDriver{}
	// Each graph is 24B, but both owners coexist. No allocation before the
	// combined 48B ceiling is admitted; borrowed edge is deliberately counted.
	if p, err := NewResidentTapeHandoff(context.Background(), d, source, target, edges, nil, nil, 47); err == nil || p != nil || d.constructions != 0 {
		t.Fatal("combined budget", err)
	}
	p, err := NewResidentTapeHandoff(context.Background(), d, source, target, edges, nil, nil, 48)
	if err != nil {
		t.Fatal(err)
	}
	if p.Close() != nil {
		t.Fatal("unrun producer close")
	}
}
func TestTrainingTapeHandoffFrozenBoundInputs(t *testing.T) {
	source, target, edges := handoffFixture()
	d := &handoffHostDriver{}
	external := &residentHostBuffer{values: []float32{10, 20}, driver: &d.residentHostDriver}
	bindings := map[int]ResidentBuffer{1: external}
	p, err := NewResidentTapeHandoff(context.Background(), d, source, target, edges, nil, bindings, 1024)
	if err != nil {
		t.Fatal(err)
	}
	bindings[1] = &residentHostBuffer{values: []float32{99, 99}}
	if p.RunDevice(context.Background()) != nil {
		t.Fatal("run")
	}
	output, err := p.OutputBuffer(2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(output.(*residentHostBuffer).values[0]-22.5)) > 2e-6 {
		t.Fatal("binding map borrowed")
	}
	// Returned consumer output may feed explicit accumulation while both owners
	// remain alive. No implicit gradient propagation crosses the producer edge.
	dst := &residentHostBuffer{values: make([]float32, 2), driver: &d.residentHostDriver}
	previous := &residentHostBuffer{values: []float32{1, 2}, driver: &d.residentHostDriver}
	if err := d.Accumulate(context.Background(), dst, previous, output, 2, .5); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(dst.values[0]-12.25)) > 2e-6 || d.owners[0].closed {
		t.Fatal("consumer accumulation lifetime")
	}
	if p.Close() != nil {
		t.Fatal("close")
	}
}

func TestTrainingTapeHandoffExternalFailure(t *testing.T) {
	source, target, edges := handoffFixture()
	d := &handoffHostDriver{}
	p, err := NewResidentTapeHandoff(context.Background(), d, source, target, edges, nil, nil, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if p.RunDevice(context.Background()) != nil {
		t.Fatal("run")
	}
	output, err := p.OutputBuffer(2)
	if err != nil {
		t.Fatal(err)
	}
	d.failure = "external-accumulation"
	dst := &residentHostBuffer{values: make([]float32, 2), driver: &d.residentHostDriver}
	previous := &residentHostBuffer{values: []float32{1, 2}, driver: &d.residentHostDriver}
	sentinel := d.Accumulate(context.Background(), dst, previous, output, 2, .5)
	if sentinel == nil {
		t.Fatal("missing external accumulation failure")
	}
	if p.RetainFailure(nil) != nil || p.RetainFailure(sentinel) != sentinel || p.terminal != sentinel || p.Close() == nil || d.owners[0].closed || d.owners[1].closed {
		t.Fatal("external output owners released")
	}
	if p.RetainFailure(errors.New("later")) == nil || p.terminal != sentinel {
		t.Fatal("original terminal overwritten")
	}
	if (*ResidentTapeHandoff)(nil).RetainFailure(sentinel) != sentinel {
		t.Fatal("nil error relay")
	}
	outputs, downloads, runs := d.outputs, d.downloads, d.runs
	if _, err := p.OutputBuffer(2); err == nil {
		t.Fatal("external terminal output")
	}
	if _, err := p.DownloadOutputs(context.Background(), []int{2}); err == nil {
		t.Fatal("external terminal download")
	}
	if p.RunDevice(context.Background()) == nil || d.outputs != outputs || d.downloads != downloads || d.runs != runs {
		t.Fatal("external terminal access reached driver")
	}
}
func TestTrainingTapeHandoffDownloadAdmission(t *testing.T) {
	source, target, edges := handoffFixture()
	d := &handoffHostDriver{}
	p, err := NewResidentTapeHandoff(context.Background(), d, source, target, edges, nil, nil, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.DownloadOutputs(context.Background(), []int{2}); err == nil {
		t.Fatal("premature download")
	}
	if p.RunDevice(context.Background()) != nil {
		t.Fatal("run")
	}
	if _, err := p.DownloadOutputs(nil, []int{2}); err == nil {
		t.Fatal("nil context")
	}
	if p.Close() != nil {
		t.Fatal("close")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if owner, err := NewResidentTapeHandoff(ctx, d, source, target, edges, nil, nil, 1024); err == nil || owner != nil {
		t.Fatal("construction cancelled")
	}
}

func TestTrainingTapeHandoffF64Reference(t *testing.T) {
	const width = 64
	ctx := context.Background()
	state := uint32(17)
	next := func() float32 {
		state = state*1664525 + 1013904223
		return (float32(state>>8)/16777216 - .5) * .5
	}
	var total, sum, squares, maximum float64
	for trial := 0; trial < 6; trial++ {
		a, b, c := make([]float32, width), make([]float32, width), make([]float32, width)
		for i := range a {
			a[i], b[i], c[i] = next(), next(), next()
		}
		source := Tape{Tensors: []TensorSpec{{Shape: []int{width}, Values: a}, {Shape: []int{width}, Values: b}, {Shape: []int{width}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: .7}}, Outputs: []int{2}}
		target := Tape{Tensors: []TensorSpec{{Shape: []int{width}, Values: make([]float32, width)}, {Shape: []int{width}, Values: c}, {Shape: []int{width}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1.3}}, Outputs: []int{2}}
		d := &handoffHostDriver{}
		p, err := NewResidentTapeHandoff(ctx, d, source, target, []TapeHandoffEdge{{Output: 2, Input: 0}}, nil, nil, width*4*6)
		if err != nil {
			t.Fatal(err)
		}
		if err = p.RunDevice(ctx); err != nil {
			t.Fatal(err)
		}
		output, err := p.OutputBuffer(2)
		if err != nil {
			t.Fatal(err)
		}
		previous := &residentHostBuffer{values: make([]float32, width), driver: &d.residentHostDriver}
		dst := &residentHostBuffer{values: make([]float32, width), driver: &d.residentHostDriver}
		for i := range previous.values {
			previous.values[i] = .1
		}
		if err = d.Accumulate(ctx, dst, previous, output, width, .5); err != nil {
			t.Fatal(p.RetainFailure(err))
		}
		for i, got := range output.(*residentHostBuffer).values {
			// Independent F64 arithmetic includes all two-tape F32 rounding error.
			want := float64(a[i]) + float64(float32(.7))*float64(b[i]) + float64(float32(1.3))*float64(c[i])
			for _, difference := range []float64{float64(got) - want, float64(dst.values[i]) - (float64(float32(.1)) + .5*want)} {
				error := math.Abs(difference)
				if error > 2e-6 {
					t.Fatalf("F64 error %.9g > 2e-6", error)
				}
				total++
				sum += error
				squares += error * error
				maximum = math.Max(maximum, error)
			}
		}
		if d.downloads != 0 || d.owners[0].closed || d.owners[1].closed {
			t.Fatal("reference used host handoff or expired owners")
		}
		if err = p.Close(); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("F64 comparisons %.0f mean_abs %.9g RMS %.9g max_abs %.9g unchanged absolute gate 2e-6", total, sum/total, math.Sqrt(squares/total), maximum)
}
