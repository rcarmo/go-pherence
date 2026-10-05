//go:build linux

package pockettts

import (
	"os"
	"testing"
)

func TestPocketReleasedJointMaximumPeripheralStages(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKET_JOINT_MAX_PERIPHERAL_AUDIT") != "1" {
		t.Skip("explicit joint maximum staged peripheral admission")
	}
	lm, flow, w, _ := releasedTrainingModels(t)
	const capBytes = int64(2 << 30)
	batch := FlowLMTrainingBatch{Frames: 375, VoiceFrames: 62, TextTokens: make([]uint32, 512), NormalizedLatents: make([]float32, 375*lm.LatentDim), VoiceLatents: make([]float32, 62*lm.LatentDim)}
	params, err := fullParameterMap(lm, flow, w)
	if err != nil {
		t.Fatal(err)
	}
	owner := &ResidentTrainingBackend{MaxResidentBytes: capBytes, MaxTapeBytes: capBytes}
	for name, values := range params {
		owner.host.Params = append(owner.host.Params, NamedTrainingTensor{Name: name, Values: values})
	}
	for i := range batch.NormalizedLatents {
		batch.NormalizedLatents[i] = .01 * float32(1+i%11)
	}
	for i := range batch.VoiceLatents {
		batch.VoiceLatents[i] = .01 * float32(1+i%7)
	}
	for i := range batch.TextTokens {
		batch.TextTokens[i] = uint32(i + 1)
	}
	rows := 1 + batch.Frames + batch.VoiceFrames + len(batch.TextTokens)
	seed := make([]float32, rows*lm.Hidden)
	for i := range seed {
		seed[i] = .001 * float32(1+i%7)
	}
	conditioner, err := lm.buildConditionerTrainingTape(batch, seed, capBytes, true)
	if err != nil {
		t.Fatal(err)
	}
	forward := resourceForwardShared(t, conditioner.Tape, conditioner.ResidentBindings, conditioner.Tape.Outputs[:1], capBytes)
	if err = owner.admitStagedStorage(conditioner.Tape, conditioner.ResidentBindings, conditioner.Tape.Outputs, false); err != nil {
		t.Fatal("production conditioner admission", err)
	}
	reverse := resourceForwardShared(t, conditioner.Tape, conditioner.ResidentBindings, conditioner.Tape.Outputs, capBytes)
	mask := make([]bool, batch.Frames)
	for i := range mask {
		mask[i] = i < batch.Frames-1
	}
	hidden := make([]float32, batch.Frames*lm.Hidden)
	for i := range hidden {
		hidden[i] = .01 * float32(1+i%11)
	}
	eos, err := lm.buildEOSProjectionTrainingTape(hidden, mask, .1, capBytes, true)
	if err != nil {
		t.Fatal(err)
	}
	if err = owner.admitStagedStorage(eos.Tape, eos.ResidentBindings, eos.Tape.Outputs, false); err != nil {
		t.Fatal("production EOS admission", err)
	}
	eosOwned := resourceForwardShared(t, eos.Tape, eos.ResidentBindings, eos.Tape.Outputs, capBytes)
	in := StepReductionInputs{Frames: batch.Frames, Hidden: lm.Hidden, Latent: lm.LatentDim, Mask: mask, DiagonalCondition: make([]float32, batch.Frames*lm.Hidden), DistillCondition: make([]float32, batch.Frames*lm.Hidden), DiagonalTarget: make([]float32, batch.Frames*lm.LatentDim), DistillTarget: make([]float32, batch.Frames*lm.LatentDim), DiagonalLogvar: make([]float32, batch.Frames), DistillLogvar: make([]float32, batch.Frames), RowMetrics: make([]float32, batch.Frames*4), EOS: .1}
	samples := WeightSampleInputs{DiagonalTime: make([]float32, batch.Frames), DistillS: make([]float32, batch.Frames), DistillT: make([]float32, batch.Frames), DiagonalSeed: in.DiagonalLogvar, DistillSeed: in.DistillLogvar}
	step, err := w.buildStepWeightTrainingTape(in, DefaultTrainingStepConfig(), samples, capBytes, true)
	if err != nil {
		t.Fatal(err)
	}
	if err = owner.admitStagedStorage(step.Tape, step.ResidentBindings, step.Tape.Outputs, false); err != nil {
		t.Fatal("production weight admission", err)
	}
	stepOwned := resourceForwardShared(t, step.Tape, step.ResidentBindings, step.Tape.Outputs, capBytes)
	largest := max(forward, reverse, eosOwned, stepOwned)
	const resident = int64(1822745808)
	t.Logf("joint peripheral conditioner source=%d forward owned=%d reverse owned=%d EOS source=%d owned=%d weight source=%d owned=%d resident+max=%d", deviceAuditTapeBytes(t, conditioner.Tape), forward, reverse, deviceAuditTapeBytes(t, eos.Tape), eosOwned, deviceAuditTapeBytes(t, step.Tape), stepOwned, resident+largest)
	if resident+largest >= capBytes {
		t.Fatal("peripheral stage joint native ceiling")
	}
	_ = flow
}
