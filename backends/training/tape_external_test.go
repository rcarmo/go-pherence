package training

import (
	"context"
	"math"
	"reflect"
	"testing"
)

func externalFixture() Tape {
	return Tape{Tensors: []TensorSpec{{Shape: []int{2}, External: true}, {Shape: []int{2}, Values: []float32{.5, .25}}, {Shape: []int{2}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: .7}}, Outputs: []int{2}}
}
func TestTrainingExternalTapeContract(t *testing.T) {
	source := externalFixture()
	if err := source.Validate(24); err != nil {
		t.Fatal(err)
	}
	if source.Validate(23) == nil {
		t.Fatal("external logical bytes excluded")
	}
	if ValidateExternalTapeBindings(source, map[int]int{}) == nil {
		t.Fatal("missing binding")
	}
	if err := ValidateExternalTapeBindings(source, map[int]int{0: 1}); err != nil {
		t.Fatal(err)
	}
	for _, id := range []int{-1, 2, 3} {
		if ValidateExternalTapeBindings(source, map[int]int{0: 1, id: 1}) == nil {
			t.Fatal("writable/invalid binding")
		}
	}
	for _, kind := range []string{"values", "write", "uninitialised", "metadata"} {
		p := copyHandoffTape(source)
		switch kind {
		case "values":
			p.Tensors[0].Values = []float32{0, 0}
		case "write":
			p.Commands[0].Output = 0
		case "uninitialised":
			p.Tensors[0].External = false
		case "metadata":
			p.Tensors[0].Shape = []int{0}
		}
		if p.Validate(1024) == nil {
			t.Fatal("invalid external source", kind)
		}
	}
}
func TestTrainingExternalTapeConversionAndPrune(t *testing.T) {
	_, source, _ := handoffFixture()
	before := copyHandoffTape(source)
	converted, err := WithExternalTapeInputs(source, []int{0}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, source) || converted.Tensors[0].Values != nil || !converted.Tensors[0].External || !reflect.DeepEqual(source.Commands, converted.Commands) {
		t.Fatal("conversion ownership/arithmetic")
	}
	converted.Tensors[0].Shape[0] = 1
	if source.Tensors[0].Shape[0] != 2 {
		t.Fatal("shape borrowed")
	}
	converted, err = WithExternalTapeInputs(source, []int{0}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	compact, mapping, err := PruneTape(converted, []int{2}, 1024)
	if err != nil || !compact.Tensors[mapping[0]].External || compact.Tensors[mapping[0]].Values != nil {
		t.Fatal("pruning erased declaration", err)
	}
	plan, err := PlanZeroTapeStorage(compact, 1024, nil)
	if err != nil || plan.Slots[mapping[0]] == plan.Slots[mapping[2]] {
		t.Fatal("external storage reused", err)
	}
	for _, kind := range []string{"none", "duplicate", "computed", "negative", "nonfinite"} {
		p := copyHandoffTape(source)
		ids := []int{0}
		switch kind {
		case "none":
			ids = nil
		case "duplicate":
			ids = []int{0, 0}
		case "computed":
			ids = []int{2}
		case "negative":
			ids = []int{-1}
		case "nonfinite":
			p.Tensors = append(p.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
		}
		if _, err := WithExternalTapeInputs(p, ids, 1024); err == nil {
			t.Fatal("bad conversion", kind)
		}
	}
}
func TestTrainingExternalTapeResidentChild(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	root, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	source := externalFixture()
	child, err := root.BeginChildTape(ctx, source, nil, []int{2}, []ResidentLeaseEdge{{Source: root, Output: 2, Input: 0}})
	if err != nil {
		t.Fatal(err)
	}
	if d.downloads != 0 || d.owners[1].buffers[0] != d.owners[0].buffers[root.outputs[2]] || !d.owners[1].tape.Tensors[0].External {
		t.Fatal("external source not bound")
	}
	values, err := child.DownloadOutputs(ctx, []int{2})
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float64{1.65, 1.475} {
		if math.Abs(float64(values[0][i])-want) > 2e-6 {
			t.Fatal("external device math", values)
		}
	}
	consumer := externalFixture()
	if _, err := child.RunSourcesTape(ctx, consumer, nil, nil, []int{2}, []ResidentLeaseEdge{{Source: child, Output: 2, Input: 0}}, false, 1); err != nil {
		t.Fatal(err)
	}
	if child.Close() != nil || root.Close() != nil || s.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingExternalTapeUnusedAdmission(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	root, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	source := externalFixture()
	source.Tensors = append(source.Tensors, TensorSpec{Shape: []int{1}, External: true})
	before := d.constructions
	if _, err := root.BeginChildTape(ctx, source, nil, []int{2}, []ResidentLeaseEdge{{Source: root, Output: 2, Input: 0}}); err == nil || d.constructions != before || s.terminal == nil {
		t.Fatal("dead missing external hidden by prune")
	}
	if root.Close() == nil {
		t.Fatal("late admission released producer")
	}
}
func TestTrainingExternalTapeOrdinaryReject(t *testing.T) {
	s, d := requireBoundResident(t)
	if _, err := s.RunUnboundTape(context.Background(), externalFixture(), []int{2}, 1024); err == nil || d.retained != nil {
		t.Fatal("unbound external allocated")
	}
	if _, err := s.RunPrunedUnboundTape(context.Background(), externalFixture(), []int{2}, 1024); err == nil {
		t.Fatal("prune admitted missing external")
	}
	if s.Close() != nil {
		t.Fatal("pure preflight terminal")
	}
}

func TestTrainingExternalTapeMetadataConstants(t *testing.T) {
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{0, 1}}, {Shape: []int{2}}}, Commands: []Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Rows: 2, Width: 1, Parts: 2, Mode: 0}}, Outputs: []int{2}}
	if tape.Validate(1024) != nil {
		t.Fatal("fixture")
	}
	// Metadata required by host admission must stay explicit. Converting it into
	// an unverified device numeric source must not bypass index validation.
	if _, err := WithExternalTapeInputs(tape, []int{1}, 1024); err == nil {
		t.Fatal("device embedding indices bypassed admission")
	}
}
func TestTrainingExternalTapeReshapeAndPooling(t *testing.T) {
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{2}, External: true}, {Shape: []int{1, 2}}, {Shape: []int{2}, Values: []float32{0, 0}}, {Shape: []int{2}, Values: []float32{0, 0}}, {Shape: []int{2}}}, Commands: []Command{{Op: "reshape", Inputs: []int{0}, Output: 1}, {Op: "vector", Inputs: []int{2, 3}, Output: 4, Scale: 1}}, Outputs: []int{1, 4}}
	plan, err := PlanZeroTapeStorage(tape, 1024, nil)
	if err != nil {
		t.Fatal(err)
	}
	if plan.Slots[0] != plan.Slots[1] || plan.Slots[0] == plan.Slots[2] || plan.Slots[2] != plan.Slots[3] || plan.Slots[0] == plan.Slots[4] {
		t.Fatal("external pooling/lifetime family")
	}
	compact, mapping, err := PruneTape(tape, []int{1}, 1024)
	if err != nil || len(compact.Commands) != 1 || !compact.Tensors[mapping[0]].External {
		t.Fatal("external reshape prune", err)
	}
}
func TestTrainingExternalTapeDeclaredRoot(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	p, b := leaseProducerFixture()
	root, err := s.BeginPrunedTapeLease(ctx, p, b, []int{2}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{2}, External: true}}, Outputs: []int{0}}
	child, err := root.BeginChildTape(ctx, tape, nil, []int{0}, []ResidentLeaseEdge{{Source: root, Output: 2, Input: 0}})
	if err != nil {
		t.Fatal(err)
	}
	if len(d.owners[1].tape.Commands) != 0 || d.owners[1].buffers[0] != d.owners[0].buffers[root.outputs[2]] || d.downloads != 0 {
		t.Fatal("external constant-only root")
	}
	if child.Close() != nil || root.Close() != nil || s.Close() != nil {
		t.Fatal("close")
	}
}

func TestTrainingExternalTapeF64Reference(t *testing.T) {
	ctx := context.Background()
	s, d := requireLeaseResident(t)
	values := make([]float32, 64)
	state := uint32(871)
	for i := range values {
		state = state*1664525 + 1013904223
		values[i] = float32(state>>8)/16777216 - .5
	}
	producer := Tape{Tensors: []TensorSpec{{Shape: []int{64}, Values: values}}, Outputs: []int{0}}
	root, err := s.BeginPrunedTapeLease(ctx, producer, nil, []int{0}, 8192)
	if err != nil {
		t.Fatal(err)
	}
	var count, sum, squares, maximum float64
	for trial := 0; trial < 8; trial++ {
		constant := make([]float32, 64)
		for i := range constant {
			constant[i] = float32(i%9) * .01
		}
		scale := float32(.1 * float32(trial))
		consumer := Tape{Tensors: []TensorSpec{{Shape: []int{64}, External: true}, {Shape: []int{64}, Values: constant}, {Shape: []int{64}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: scale}}, Outputs: []int{2}}
		got, err := root.RunSourcesTape(ctx, consumer, nil, nil, []int{2}, []ResidentLeaseEdge{{Source: root, Output: 0, Input: 0}}, false, 1)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got[0] {
			want := float64(values[i]) + float64(scale)*float64(constant[i])
			delta := math.Abs(float64(v) - want)
			if delta > 2e-6 {
				t.Fatal("F64 difference", delta)
			}
			count++
			sum += delta
			squares += delta * delta
			maximum = math.Max(maximum, delta)
		}
	}
	if d.executions != 9 || d.downloads != 8 || d.accumulations != 0 {
		t.Fatal("download/gradient contract")
	}
	if root.Close() != nil || s.Close() != nil {
		t.Fatal("close")
	}
	t.Logf("F64 comparisons %.0f mean_abs %.9g RMS %.9g max_abs %.9g unchanged absolute gate 2e-6", count, sum/count, math.Sqrt(squares/count), maximum)
}
