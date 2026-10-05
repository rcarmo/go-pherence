package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func stepBackboneFixture(t *testing.T, voice int) (*FlowLMTrainingCPU, *LSDWeightMLP, FlowLMTrainingBatch, []float32, StepReductionInputs, WeightSampleInputs) {
	t.Helper()
	o := loadTrainingStepOracle(t)
	lm, _, w := trainingStepModelsFromOracle(t, o)
	frames := 2
	batch := FlowLMTrainingBatch{Frames: frames, VoiceFrames: voice, NormalizedLatents: make([]float32, frames*lm.LatentDim), VoiceLatents: make([]float32, voice*lm.LatentDim), TextTokens: []uint32{1, 1}}
	in := StepReductionInputs{Frames: frames, Hidden: lm.Hidden, Latent: lm.LatentDim, Mask: []bool{true, false}, DiagonalCondition: make([]float32, frames*lm.Hidden), DistillCondition: make([]float32, frames*lm.Hidden), DiagonalTarget: make([]float32, frames*lm.LatentDim), DistillTarget: make([]float32, frames*lm.LatentDim), DiagonalLogvar: []float32{.2, 0}, DistillLogvar: []float32{-.1, 0}, RowMetrics: []float32{.1, .2, .3, .4, 0, 0, 0, 0}, EOS: .25}
	for i := 0; i < lm.Hidden; i++ {
		in.DiagonalCondition[i] = float32(i+1) * .03
		in.DistillCondition[i] = float32(i+1) * .02
	}
	for i := 0; i < lm.LatentDim; i++ {
		in.DiagonalTarget[i] = float32(i+1) * .05
		in.DistillTarget[i] = float32(i+1) * .04
	}
	seed := make([]float32, (1+voice+2+frames)*lm.Hidden)
	p, err := lm.BuildBackboneTrainingTape(batch, seed, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	v := interpretFlowTape(t, p.Tape)
	hidden := append([]float32(nil), v[1][len(seed)-frames*lm.Hidden:]...)
	return lm, w, batch, hidden, in, stepWeightSamples(in)
}
func TestTrainingStepBackboneTapeReference(t *testing.T) {
	for _, voice := range []int{0, 1} {
		for _, p := range []float32{0, .25, 1} {
			lm, w, batch, hidden, in, s := stepBackboneFixture(t, voice)
			cfg := TrainingStepConfig{PEqual: p, EOSLossWeight: .1}
			step, err := w.BuildStepWeightTrainingTape(in, cfg, s, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			sv := interpretFlowTape(t, step.Tape)
			rows := 1 + batch.VoiceFrames + len(batch.TextTokens) + batch.Frames
			seed := make([]float32, rows*lm.Hidden)
			copy(seed[len(seed)-len(sv[0]):], sv[0])
			back, err := lm.BuildEOSBackboneTrainingTape(batch, seed, hidden, in.Mask, cfg.EOSLossWeight, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			bv := interpretFlowTape(t, back.Tape)
			index := 2
			if voice > 0 {
				index++
			}
			final := make([]float32, len(sv[1]))
			for i := range final {
				final[i] = bv[index][i] + sv[1][i]
			}
			plan, err := lm.BuildStepBackboneTrainingTape(w, batch, hidden, in, cfg, s, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			got := interpretFlowTape(t, plan.Tape)
			for i := 0; i < 5; i++ {
				assertSliceClose(t, "public reduction", got[i], sv[i], 2e-6)
			}
			outIndex := 5
			if voice > 0 {
				assertSliceClose(t, "voice", got[outIndex], bv[2], 1.5e-5)
				outIndex++
			}
			assertSliceClose(t, "final normalized", got[outIndex], final, 1.5e-5)
			expected := map[string][]float32{}
			for i, id := range step.Tape.Outputs {
				for name, b := range step.ResidentBindings {
					if b.Gradient == id {
						expected[name] = sv[i]
					}
				}
			}
			for i, id := range back.Tape.Outputs {
				for name, b := range back.ResidentBindings {
					if b.Gradient == id {
						expected[name] = bv[i]
					}
				}
			}
			for i, id := range plan.Tape.Outputs {
				for name, b := range plan.ResidentBindings {
					if b.Gradient == id {
						assertSliceClose(t, name, got[i], expected[name], 2e-5)
					}
				}
			}
			if len(plan.ResidentBindings) != len(step.ResidentBindings)+len(back.ResidentBindings) {
				t.Fatal("allparams")
			}
			if err := train.ValidateResidentTapeBindings(plan.Tape, plan.ResidentBindings, 64<<20); err != nil {
				t.Fatal(err)
			}
		}
	}
}
func TestTrainingStepBackboneTapeAdmissionOwnership(t *testing.T) {
	lm, w, batch, hidden, in, s := stepBackboneFixture(t, 1)
	cfg := DefaultTrainingStepConfig()
	p, err := lm.BuildStepBackboneTrainingTape(w, batch, hidden, in, cfg, s, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, x := range p.Tape.Tensors {
		n, _ := x.Elements()
		bytes += int64(n) * 4
	}
	if _, err := lm.BuildStepBackboneTrainingTape(w, batch, hidden, in, cfg, s, bytes-1); err == nil {
		t.Fatal("combined ceiling")
	}
	if _, err := lm.BuildStepBackboneTrainingTape(w, batch, hidden, in, cfg, s, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"topology", "frames", "rows", "weight", "hidden", "seed", "budget"} {
		t.Run(kind, func(t *testing.T) {
			lm, w, batch, hidden, in, s := stepBackboneFixture(t, 1)
			budget := int64(64 << 20)
			switch kind {
			case "topology":
				in.Hidden++
			case "frames":
				in.Frames++
			case "rows":
				batch.TextTokens = make([]uint32, 1024)
			case "weight":
				w = nil
			case "hidden":
				hidden[0] = float32(math.NaN())
			case "seed":
				s.DiagonalSeed[0]++
			case "budget":
				budget = 0
			}
			if _, err := lm.BuildStepBackboneTrainingTape(w, batch, hidden, in, cfg, s, budget); err == nil {
				t.Fatal("bad source")
			}
		})
	}
	before := interpretFlowTape(t, p.Tape)
	for _, x := range [][]float32{lm.Input.Weight, lm.Transformer.Layers[0].FC1.Weight, w.Layers[0].Weight, batch.NormalizedLatents, batch.VoiceLatents, in.DiagonalCondition, in.RowMetrics, s.DiagonalTime} {
		x[0] = 99
	}
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
}
func TestTrainingStepBackboneBoundaryGuards(t *testing.T) {
	lm, w, batch, hidden, in, s := stepBackboneFixture(t, 1)
	p, err := lm.BuildStepBackboneTrainingTape(w, batch, hidden, in, DefaultTrainingStepConfig(), s, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	fresh := func() [][]float32 { return interpretFlowTape(t, p.Tape)[:7] }
	if _, err := decodeStepBackboneBoundary(fresh(), batch, hidden, in, s); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"count", "shape", "nan", "overlap", "source", "tail"} {
		v := fresh()
		switch kind {
		case "count":
			v = v[:6]
		case "shape":
			v[5] = nil
		case "nan":
			v[6][0] = float32(math.NaN())
		case "overlap":
			v[6] = v[1]
		case "source":
			v[5] = batch.VoiceLatents
		case "tail":
			v[6][len(v[6])-1] += 1
		}
		if _, err := decodeStepBackboneBoundary(v, batch, hidden, in, s); err == nil {
			t.Fatal(kind)
		}
	}
}
