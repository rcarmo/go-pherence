package pockettts

import "fmt"

// Step-only capability: sample times are frozen and the step discards the MLP
// input VJP and repeated forward output. Public seeded APIs keep both outputs.
type trainingWeightRootsBackend interface {
	weightRootsEnabled() bool
	accumulateSampleWeight(*LSDWeightMLP, WeightSampleInputs) error
}

func (b *ResidentTrainingBackend) weightRootsEnabled() bool {
	return b != nil && (b.stepWeightEnabled() || b.Scope == "ptx-resident-weightroots-hybrid-v1" || b.Scope == "vulkan-resident-weightroots-hybrid-v1")
}

func (b *ResidentTrainingBackend) accumulateSampleWeight(m *LSDWeightMLP, samples WeightSampleInputs) error {
	if err := b.admitGraph(); err != nil {
		return err
	}
	if !b.weightRootsEnabled() || b.trainer.Weighting != m || m.TrainingBackend != b {
		return b.fail(fmt.Errorf("sample weight roots owner/scope"))
	}
	if len(samples.DiagonalTime) < 1 || len(samples.DiagonalTime) > 512 {
		return b.fail(fmt.Errorf("sample weight roots frame ceiling"))
	}
	plan, err := m.buildWeightTrainingTape(nil, nil, 2*len(samples.DiagonalTime), b.MaxTapeBytes, &samples, true)
	if err != nil {
		return b.fail(err)
	}
	// Full source/binding admission is unchanged. Pruning roots every accumulated
	// parameter gradient, even with zero seeds; no discarded boundary is requested.
	_, err = b.runResidentTape(plan.Tape, plan.ResidentBindings, nil, true, 1)
	return b.fail(err)
}
