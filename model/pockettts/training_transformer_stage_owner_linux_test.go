//go:build linux

package pockettts

import (
	"context"
	"sort"
	"testing"
)

func TestTrainingTransformerStageOwnerReference(t *testing.T) {
	r, b, d := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	m := r.trainer.FlowLM.Transformer
	rows := 3
	sequence, seed := make([]float32, rows*m.Width), make([]float32, rows*m.Width)
	for i := range sequence {
		sequence[i] = float32(i%11-5) * .03
		seed[i] = float32(i%7-3) * .02
	}
	want, grads, dx, err := m.ForwardBackward(sequence, seed, rows)
	if err != nil {
		t.Fatal(err)
	}
	output, err := b.beginTransformerStages(m, sequence, rows)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "stage owner output", output, want, 4e-6)
	output[0] = 99 // exported result cannot mutate private paired boundaries
	got, err := b.backwardTransformerStages(m, sequence, seed, rows)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "stage owner input VJP", got, dx, 8e-6)
	expected := map[string][]float32{}
	addFlowLMGradients(expected, &FlowLMTrainingGradients{Transformer: grads})
	captured := map[string][]float32{}
	// Driver records gradient values during Update preflight. Touch unrelated
	// resident parameters with explicit zero arrays to isolate this stage test.
	for _, binding := range r.trainer.bindings {
		if x, ok := expected[binding.name]; ok && len(x) > 0 {
			captured[binding.name] = x
		} else {
			if err := b.state.Accumulate(b.ctx, map[string][]float32{binding.name: make([]float32, len(binding.parameter))}, 1); err != nil {
				t.Fatal(err)
			}
		}
	}
	if _, err = b.state.Update(b.ctx, 100000); err != nil {
		t.Fatal(err)
	}
	// Resident entries are sorted by name, so captured gradient order is sorted.
	names := make([]string, 0, len(r.trainer.params))
	for name := range r.trainer.params {
		names = append(names, name)
	}
	sort.Strings(names)
	for i, name := range names {
		if want, ok := captured[name]; ok {
			assertSliceClose(t, "stage owner gradient "+name, d.validatedGradients[i], want, 1e-5)
		}
	}
	if b.transformerStages != nil {
		t.Fatal("paired stage retained after success")
	}
	if err = b.state.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestTrainingTransformerStageOwnerFailure(t *testing.T) {
	for _, kind := range []string{"forward-failure", "boundary-change", "parameter-change", "backward-failure", "update-before-backward"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			m := r.trainer.FlowLM.Transformer
			rows := 3
			x, seed := make([]float32, rows*m.Width), make([]float32, rows*m.Width)
			if kind == "forward-failure" {
				d.failTape = 1
			}
			_, err := b.beginTransformerStages(m, x, rows)
			if kind != "forward-failure" {
				if err != nil {
					t.Fatal(err)
				}
				switch kind {
				case "boundary-change":
					x[0] = 1
				case "parameter-change":
					m.Layers[0].FC1.Weight[0] += 1
				case "backward-failure":
					d.failTape = d.tapes + 1
				case "update-before-backward":
					err = b.UpdateTrainer(b.ctx, r.trainer, TrainingStepGradients{}, 1)
				}
				if kind != "update-before-backward" {
					_, err = b.backwardTransformerStages(m, x, seed, rows)
				}
			}
			if err == nil || b.terminal == nil {
				t.Fatal("stage failure not fenced", kind, err)
			}
			calls := d.tapes
			if _, err = b.beginTransformerStages(m, x, rows); err == nil || d.tapes != calls {
				t.Fatal("stage retry")
			}
			if b.Close() == nil {
				t.Fatal("failed stage owner closed")
			}
		})
	}
}
