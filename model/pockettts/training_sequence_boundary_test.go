//go:build linux

package pockettts

import (
	"context"
	"testing"
)

func TestTrainingResidentSequenceEmptyTextBoundary(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	cpu, _, _ := trainingStepModelsFromOracle(t, oracle)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	backend, d := installResidentGraph(t, trainer, "ptx-resident-sequence-hybrid-v1")
	if err := backend.PrepareTrainingStep(context.Background(), trainer); err != nil {
		t.Fatal(err)
	}
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: 0, NormalizedLatents: oracle.NormalizedLatents}
	dz, deos := make([]float32, batch.Frames*lm.Hidden), make([]float32, batch.Frames)
	for i := range dz {
		dz[i] = float32(i+1) * .01
	}
	for i := range deos {
		deos[i] = .25
	}
	wantTape, err := cpu.forwardTrainingTapeMode(batch, dz, deos, true)
	if err != nil {
		t.Fatal(err)
	}
	wantG, wantIn, err := cpu.backwardTrainingTape(wantTape, batch, dz, deos)
	if err != nil {
		t.Fatal(err)
	}
	gotTape, err := lm.forwardTrainingTapeMode(batch, dz, deos, true)
	if err != nil {
		t.Fatal(err)
	}
	g, in, err := lm.backwardTrainingTape(gotTape, batch, dz, deos)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "empty text/voice sequence", gotTape.output.Sequence, wantTape.output.Sequence, 2e-6)
	assertSliceClose(t, "empty text/voice Z", gotTape.output.Z, wantTape.output.Z, 5e-6)
	assertSliceClose(t, "empty text/voice audio input gradient", in.NormalizedLatents, wantIn.NormalizedLatents, 5e-5)
	// Device-owned sequence parameter gradients must not return on host.
	for _, name := range []string{"flow_lm.conditioner.embed.weight", "flow_lm.bos_emb", "flow_lm.bos_before_voice"} {
		for _, v := range fullGradientByName(TrainingStepGradients{FlowLM: g}, name) {
			if v != 0 {
				t.Fatal("resident gradient returned on host")
			}
		}
	}
	for _, v := range wantG.Embedding {
		if v != 0 {
			t.Fatal("empty text CPU reference gradient")
		}
	}
	// Paired backward still accumulates all transformer, sequence and projection
	// parameters, even empty embedding/speaker branches; no text ID is read.
	parameters := map[string][]float32{}
	addTransformerParameters(parameters, lm.Transformer)
	expected := 3 + 4 + len(parameters)
	if d.accumulations != expected {
		t.Fatal("empty text incomplete accumulation", d.accumulations, expected)
	}
	lm.SequenceBackend = nil
	before := d.tapes
	if _, _, err := lm.backwardTrainingTape(gotTape, batch, dz, deos); err == nil || before != d.tapes {
		t.Fatal("removed sequence backend dispatched")
	}
}
func TestTrainingSequenceRequiresProjection(t *testing.T) {
	lm := tinyFlowLMTraining()
	lm.SequenceBackend = &ResidentTrainingBackend{}
	batch := FlowLMTrainingBatch{Frames: 1, TextTokens: []uint32{0}, NormalizedLatents: make([]float32, lm.LatentDim)}
	if _, err := lm.forwardTrainingTape(batch, make([]float32, lm.Hidden), []float32{0}); err == nil {
		t.Fatal("unbound projections admitted")
	}
}
