//go:build linux

package pockettts

import (
	"context"
	"testing"
)

func TestPocketSharedTapeOnlyPreparationInventory(t *testing.T) {
	r := requireRunner(t)
	b, _ := installResidentGraph(t, r.trainer, "vulkan-resident-sharedworkspace-hybrid-v1")
	d := &sharedSnapshotOnlyDriver{}
	b.Driver = d
	if err := r.trainer.refreshParameterBindings(); err != nil {
		t.Fatal(err)
	}
	for _, binding := range r.trainer.bindings {
		if !b.ownsResidentGradient(binding.name) {
			t.Fatal("unowned shared gradient", binding.name)
		}
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if d.allocations != 5*len(b.host.Params)+6 {
		t.Fatal("shared prepare reserved host-upload branch", d.allocations, len(b.host.Params))
	}
	// A clean preparation has executed no tapes; only the new host Accumulate
	// admission is exercised. It must reject without terminal or driver action.
	p := b.host.Params[0]
	before := d.allocations
	if err := b.state.Accumulate(context.Background(), map[string][]float32{p.Name: make([]float32, len(p.Values))}, 1); err == nil {
		t.Fatal("shared backend permits host fallback")
	}
	if d.allocations != before || b.terminal != nil {
		t.Fatal("host rejection changed backend")
	}
	b.active = false
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
	legacy := &ResidentTrainingBackend{Scope: "vulkan-resident-hybrid-v1"}
	if legacy.ownsResidentGradient("flow.w_s_t.0.weight") || legacy.ownsResidentGradient("flow_lm.bos_emb") || b.ownsResidentGradient("unowned.parameter") {
		t.Fatal("ownership predicate broadened")
	}
}
