//go:build linux

package pockettts

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
)

func TestTrainingFlowDeviceInputsLease(t *testing.T) {
	ctx := context.Background()
	_, _, device, bound := deviceFlowFixture(t, 1)
	d := &hostResidentGraphDriver{t: t}
	snapshot := train.ResidentSnapshot{Config: train.ResidentConfig{LearningRate: .001, Beta1: .9, Beta2: .99, Epsilon: 1e-8, EMADecay: .9}, Parameters: map[string]train.ResidentParameter{}}
	for name, b := range device.flow.ResidentBindings {
		p := device.flow.Tape.Tensors[b.Parameter].Values
		snapshot.Parameters[name] = train.ResidentParameter{Parameter: p, M: make([]float32, len(p)), V: make([]float32, len(p)), EMA: append([]float32(nil), p...)}
	}
	state, err := train.NewResidentState(ctx, d, snapshot, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	in := primaryFixture()
	producer := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{3}, Values: in.Condition}, {Shape: []int{2}, Values: in.Input}, {Shape: []int{2}, Values: in.GV}, {Shape: []int{2}, Values: in.GT}}, Outputs: []int{0, 1, 2, 3}}
	root, err := state.BeginZeroPrunedTapeLease(ctx, producer, nil, producer.Outputs, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	edges := []train.ResidentLeaseEdge{{Source: root, Output: 0, Input: device.inputs.Condition}, {Source: root, Output: 1, Input: device.inputs.Input}, {Source: root, Output: 2, Input: device.inputs.ValueSeed}, {Source: root, Output: 3, Input: device.inputs.TangentSeed}}
	outputs := device.flow.Tape.Outputs[:device.flow.boundarySize()]
	values, err := root.RunSourcesTape(ctx, device.flow.Tape, device.flow.ResidentBindings, nil, outputs, edges, true, .25)
	if err != nil {
		t.Fatal(err)
	}
	expected, err := device.flow.Decode(interpretFlowTapeStorageBound(t, device.flow.Tape, true, bound))
	if err != nil {
		t.Fatal(err)
	}
	got, err := device.flow.decodeBoundary(values)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "lease output", got.Output, expected.Output, 2e-6)
	assertSliceClose(t, "lease tangent", got.Tangent, expected.Tangent, 2e-6)
	assertSliceClose(t, "lease input", got.DInput, expected.DInput, 2e-6)
	assertSliceClose(t, "lease condition", got.DCondition, expected.DCondition, 2e-6)
	assertSliceClose(t, "lease times", got.DTimes, expected.DTimes, 2e-6)
	if d.downloadedElements != len(got.Output)+len(got.Tangent)+len(got.DInput)+len(got.DCondition)+len(got.DTimes) || d.accumulations != len(device.flow.ResidentBindings) {
		t.Fatal("source/parameter gradients downloaded or accumulation count changed")
	}
	if root.Close() != nil {
		t.Fatal("lease close")
	}
	// The mock records device gradients at update preflight without downloading
	// them or reopening any graph. Resident parameters use sorted name order.
	if _, err := state.Update(ctx, 100); err != nil {
		t.Fatal(err)
	}
	if len(d.validatedGradients) != len(device.flow.ParameterNames) {
		t.Fatal("gradient preflight coverage")
	}
	for index, name := range device.flow.ParameterNames {
		v := expected.Parameters[name]
		want := make([]float32, len(v))
		for i, x := range v {
			want[i] = .25 * x
		}
		assertSliceClose(t, name, d.validatedGradients[index], want, 2e-6)
	}
	if state.Close() != nil {
		t.Fatal("state close")
	}
}
