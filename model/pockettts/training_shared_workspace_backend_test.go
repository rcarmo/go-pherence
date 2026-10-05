//go:build linux

package pockettts

import (
	"context"
	train "github.com/rcarmo/go-pherence/backends/training"
	"sort"
	"testing"
)

func TestPocketSharedWorkspaceScopeAdmission(t *testing.T) {
	o := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, o)
	trainer, e := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if e != nil {
		t.Fatal(e)
	}
	identity := runIdentityFixture()
	identity.Backend = "vulkan-resident-sharedworkspace-hybrid-v1"
	if e := identity.validate(); e != nil {
		t.Fatal("shared replay scope", e)
	}
	identity.Backend = "ptx-resident-sharedworkspace-hybrid-v1"
	if identity.validate() == nil {
		t.Fatal("unimplemented replay scope")
	}
	b, d := installResidentGraph(t, trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	if !b.sharedWorkspaceEnabled() || !b.deviceHandoffEnabled() || !b.ownsStateCheck() || !b.ownsDeviceScale() || !b.ownsStepReduction() {
		t.Fatal("shared scope inheritance")
	}
	if e = b.PrepareTrainingStep(context.Background(), trainer); e == nil || d.uploads != 0 || b.state != nil {
		t.Fatal("missing shared capability crossed allocation", e)
	}
	for _, scope := range []string{"vulkan-resident-devicehandoff-hybrid-v1", "ptx-resident-devicehandoff-hybrid-v1", "ptx-resident-sharedworkspace-hybrid-v1"} {
		if (&ResidentTrainingBackend{Scope: scope}).sharedWorkspaceEnabled() {
			t.Fatal("legacy/unsupported scope enabled", scope)
		}
	}
}

func TestTrainingSharedWorkspaceGroupedBackbone(t *testing.T) {
	// Exercise the exact grouped helper with the host lease interpreter, without
	// pretending that this legacy mock implements native shared-workspace.
	r, b, d := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	g := &flowTapeGraph{}
	z := g.zero(1)
	g.tape.Outputs = []int{z}
	lease, err := b.state.BeginZeroPrunedTapeLease(b.ctx, g.tape, nil, []int{z}, b.MaxTapeBytes)
	if err != nil {
		t.Fatal(err)
	}
	b.handoffLease = lease
	source := train.Tape{}
	bindings := map[string]train.ResidentTapeBinding{}
	for i, binding := range r.trainer.bindings {
		p := len(source.Tensors)
		source.Tensors = append(source.Tensors, train.TensorSpec{Shape: []int{len(binding.parameter)}, Values: binding.parameter})
		seed := len(source.Tensors)
		values := make([]float32, len(binding.parameter))
		for j := range values {
			values[j] = float32(i+j+1) * .001
		}
		source.Tensors = append(source.Tensors, train.TensorSpec{Shape: []int{len(values)}, Values: values})
		out := len(source.Tensors)
		source.Tensors = append(source.Tensors, train.TensorSpec{Shape: []int{len(values)}})
		source.Commands = append(source.Commands, train.Command{Op: "vector", Inputs: []int{p, seed}, Output: out, Scale: .25})
		source.Outputs = append(source.Outputs, out)
		bindings[binding.name] = train.ResidentTapeBinding{Parameter: p, Gradient: out}
	}
	if _, err = b.runGroupedStepBackbone(source, bindings, nil); err != nil {
		t.Fatal(err)
	}
	if err = lease.Close(); err != nil {
		t.Fatal(err)
	}
	b.handoffLease = nil
	if _, err = b.state.Update(b.ctx, 100000); err != nil {
		t.Fatal(err)
	}
	names := make([]string, 0, len(bindings))
	for name := range bindings {
		names = append(names, name)
	}
	sort.Strings(names)
	for i, binding := range r.trainer.bindings {
		want := make([]float32, len(binding.parameter))
		for j, x := range binding.parameter {
			want[j] = x + .25*float32(i+j+1)*.001
		}
		assertSliceClose(t, "grouped accumulated "+binding.name, d.validatedGradients[sort.SearchStrings(names, binding.name)], want, 6e-8)
	}
	// This direct helper test did not use the backend's Step completion path.
	// The clean resident owner can close; backend.active correctly stays fenced.
	if err = b.state.Close(); err != nil {
		t.Fatal(err)
	}
}
