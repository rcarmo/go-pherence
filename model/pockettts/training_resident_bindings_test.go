package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
)

func TestTrainingResidentModelBindings(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, _ := trainingStepModelsFromOracle(t, oracle)
	condition := make([]float32, lm.Hidden)
	noise, seed := make([]float32, lm.LatentDim), make([]float32, lm.LatentDim)
	tape, err := flow.BuildFlowDualTrainingTape(condition, []float32{.2, .8}, noise, 1, seed, seed, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(tape.ResidentBindings) != len(tape.ParameterNames) {
		t.Fatal("flow binding coverage")
	}
	if err := train.ValidateResidentTapeBindings(tape.Tape, tape.ResidentBindings, 64<<20); err != nil {
		t.Fatal(err)
	}
	for i, name := range tape.ParameterNames {
		b := tape.ResidentBindings[name]
		if tape.Tape.Outputs[4+2*len(flow.Time)+i] != b.Gradient {
			t.Fatal("flow gradient binding ordering")
		}
	}
	rows := 3
	x, dout := make([]float32, rows*lm.Hidden), make([]float32, rows*lm.Hidden)
	transformer, err := lm.Transformer.BuildTransformerTrainingTape(x, dout, rows, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(transformer.ResidentBindings) != len(transformer.ParameterNames) {
		t.Fatal("transformer binding coverage")
	}
	if err := train.ValidateResidentTapeBindings(transformer.Tape, transformer.ResidentBindings, 64<<20); err != nil {
		t.Fatal(err)
	}
	for i, name := range transformer.ParameterNames {
		b := transformer.ResidentBindings[name]
		if transformer.Tape.Outputs[2+i] != b.Gradient {
			t.Fatal("transformer gradient binding ordering")
		}
	}
}
