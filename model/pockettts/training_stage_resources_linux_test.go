//go:build linux

package pockettts

import (
	"context"
	"math"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

func TestTrainingStagedResourcePreflight(t *testing.T) {
	for _, kind := range []string{"fits", "joint-cap", "source-cap", "dead-source", "external"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			resident, err := b.stagedResidentPayload()
			if err != nil {
				t.Fatal(err)
			}
			tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{0}}, {Shape: []int{16}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 0}, Output: 1, Mode: 2}}, Outputs: []int{1}}
			// The unbound utility tape is not accepted here: stage admission requires
			// a complete named parameter inventory, even if execution would prune it.
			stage, err := buildTransformerLayerStage(r.trainer.FlowLM.Transformer, 0, make([]float32, 3*r.trainer.FlowLM.Hidden), make([]float32, 3*r.trainer.FlowLM.Hidden), 3, false, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "joint-cap":
				b.MaxResidentBytes = resident + 1
			case "source-cap":
				b.MaxTapeBytes = 1
			case "dead-source":
				stage.Tape.Tensors = append(stage.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "external":
				stage.Tape.Tensors = append(stage.Tape.Tensors, train.TensorSpec{Shape: []int{1}, External: true})
			}
			before := d.tapes
			err = b.admitStagedStorage(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[1:2], true)
			if (err == nil) != (kind == "fits") {
				t.Fatal("stage resource result", kind, err)
			}
			if d.tapes != before {
				t.Fatal("resource preflight constructed/executed native tape")
			}
			if err = b.admitStagedStorage(tape, nil, tape.Outputs, false); err == nil {
				t.Fatal("missing stage binding inventory")
			}
		})
	}
}

// Immutable-zero sharing is value-sensitive: a geometry-only zero forecast
// must not license a later nonzero seed at the same resident ceiling.
func TestTrainingStagedResourceActualSeed(t *testing.T) {
	r, b, d := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	m := r.trainer.FlowLM.Transformer
	const rows = 129
	sequence, seed := make([]float32, rows*m.Width), make([]float32, rows*m.Width)
	zero, err := buildTransformerLayerStage(m, 0, sequence, seed, rows, false, b.MaxTapeBytes)
	if err != nil {
		t.Fatal(err)
	}
	for i := range seed {
		seed[i] = .001 * float32(1+i%7)
	}
	actual, err := buildTransformerLayerStage(m, 0, sequence, seed, rows, false, b.MaxTapeBytes)
	if err != nil {
		t.Fatal(err)
	}
	owned := func(stage TransformerTrainingTape) int64 {
		return max(resourceForwardShared(t, stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[1:2], b.MaxTapeBytes), resourceGroupedBackbone(t, stage.Tape, stage.ResidentBindings, nil, b.MaxTapeBytes))
	}
	zeroOwned, actualOwned := owned(zero), owned(actual)
	if actualOwned <= zeroOwned {
		t.Fatal("fixture does not distinguish actual seed storage", zeroOwned, actualOwned)
	}
	resident, err := b.stagedResidentPayload()
	if err != nil {
		t.Fatal(err)
	}
	b.MaxResidentBytes = resident + zeroOwned + 1
	if err = b.admitStagedStorage(zero.Tape, zero.ResidentBindings, zero.Tape.Outputs[1:2], true); err != nil {
		t.Fatal(err)
	}
	if err = b.admitStagedStorage(actual.Tape, actual.ResidentBindings, actual.Tape.Outputs[1:2], true); err == nil || d.tapes != 0 {
		t.Fatal("nonzero source crossed zero forecast", err, d.tapes)
	}
}

func TestTrainingStagedResourceStopsBeforeFirstStage(t *testing.T) {
	r, b, d := residentGraphRunner(t, "vulkan-resident-devicehandoff-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	resident, err := b.stagedResidentPayload()
	if err != nil {
		t.Fatal(err)
	}
	b.MaxResidentBytes = resident + 1
	batch := r.rows[0].Batch
	mask := make([]bool, batch.Frames)
	for i := range mask {
		mask[i] = i < r.rows[0].ValidFrames
	}
	if _, _, err = b.evaluateStagedForward(r.trainer.FlowLM, batch, mask, .1); err == nil || b.terminal == nil || d.tapes != 0 {
		t.Fatal("joint payload crossed execution boundary", err, d.tapes)
	}
}
