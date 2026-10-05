package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func TestTrainingStepResultsReference(t *testing.T) {
	for _, voice := range []int{0, 1} {
		lm, w, batch, hidden, in, s := stepBackboneFixture(t, voice)
		cfg := DefaultTrainingStepConfig()
		p, err := lm.BuildStepBackboneTrainingTape(w, batch, hidden, in, cfg, s, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		full := interpretFlowTape(t, p.Tape)
		count := 6 + voice
		selected, err := buildStepResultsTape(p, batch, in, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		all := interpretFlowTape(t, selected.Tape)
		byID := map[int][]float32{}
		for i, id := range selected.Tape.Outputs {
			byID[id] = all[i]
		}
		out := [][]float32{}
		for _, id := range stepResultsOutputs(selected, batch) {
			out = append(out, byID[id])
		}
		got, err := decodeStepResultsBoundary(out, batch, hidden, in, s)
		if err != nil {
			t.Fatal(err)
		}
		if got.Reduction.DCondition != nil || got.Reduction.DTarget != nil {
			t.Fatal("fake discarded full outputs")
		}
		assertSliceClose(t, "diag seeds", got.Reduction.DiagonalLogvar, full[2], 2e-6)
		assertSliceClose(t, "dist seeds", got.Reduction.DistillLogvar, full[3], 2e-6)
		if voice > 0 {
			assertSliceClose(t, "voice", got.Inputs.VoiceLatents, full[5], 1.5e-5)
		}
		assertSliceClose(t, "normalized", got.Inputs.NormalizedLatents, full[count-1], 1.5e-5)
		var bytes int64
		for _, x := range selected.Tape.Tensors {
			n, _ := x.Elements()
			bytes += int64(n) * 4
		}
		if _, err := buildStepResultsTape(p, batch, in, bytes-1); err == nil {
			t.Fatal("tail logical byte ceiling")
		}
		if _, err := buildStepResultsTape(p, batch, in, bytes); err != nil {
			t.Fatal(err)
		}
	}
}
func TestTrainingStepResultsBoundaryGuards(t *testing.T) {
	lm, w, batch, hidden, in, s := stepBackboneFixture(t, 1)
	p, err := lm.BuildStepBackboneTrainingTape(w, batch, hidden, in, DefaultTrainingStepConfig(), s, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	p, err = buildStepResultsTape(p, batch, in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	fresh := func() [][]float32 {
		v := interpretFlowTape(t, p.Tape)
		byID := map[int][]float32{}
		for i, id := range p.Tape.Outputs {
			byID[id] = v[i]
		}
		out := [][]float32{}
		for _, id := range stepResultsOutputs(p, batch) {
			out = append(out, append([]float32(nil), byID[id]...))
		}
		return out
	}
	if _, err := decodeStepResultsBoundary(fresh(), batch, hidden, in, s); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"count", "metrics", "shape", "nan", "padding", "negative", "overlap", "source", "tail", "inactive-tail"} {
		v := fresh()
		switch kind {
		case "count":
			v = v[:4]
		case "metrics":
			v[2] = nil
		case "shape":
			v[3] = nil
		case "nan":
			v[4][0] = float32(math.NaN())
		case "padding":
			v[0][1] = 1
		case "negative":
			v[2][0] = -1
		case "overlap":
			v[5] = v[4][len(v[4])-in.Latent:]
		case "source":
			v[3] = batch.VoiceLatents
		case "tail":
			v[4][len(v[4])-1]++
		case "inactive-tail":
			v[5][0] = 1
			v[4][len(v[4])-in.Latent] = 1
		}
		if _, err := decodeStepResultsBoundary(v, batch, hidden, in, s); err == nil {
			t.Fatal(kind)
		}
	}
}

func TestTrainingStepResultsSourceAdmission(t *testing.T) {
	lm, w, batch, hidden, in, s := stepBackboneFixture(t, 1)
	p, err := lm.BuildStepBackboneTrainingTape(w, batch, hidden, in, DefaultTrainingStepConfig(), s, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	badBatch := batch
	badBatch.Frames = 0
	if _, err := buildStepResultsTape(p, badBatch, in, 64<<20); err == nil {
		t.Fatal("bad tail source topology")
	}
	p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if _, err := buildStepResultsTape(p, batch, in, 64<<20); err == nil {
		t.Fatal("invalid dead full source hidden by result selection")
	}
}
