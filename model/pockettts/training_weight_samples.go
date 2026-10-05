package pockettts

import "fmt"

// WeightSampleInputs retains separate frozen sample columns and already-scaled
// loss seeds. The tape packs diagonal rows before distillation rows. It returns
// the MLP input VJP, not derivatives through packing back into sample columns.
type WeightSampleInputs struct {
	DiagonalTime, DistillS, DistillT []float32
	DiagonalSeed, DistillSeed        []float32
}

func (s WeightSampleInputs) validate(rows int) error {
	if rows < 2 || rows > 1024 || rows%2 != 0 {
		return fmt.Errorf("sample weighting row ceiling")
	}
	for _, x := range [][]float32{s.DiagonalTime, s.DistillS, s.DistillT, s.DiagonalSeed, s.DistillSeed} {
		if len(x) != rows/2 || !finiteF32(x) {
			return fmt.Errorf("sample weighting column shape/nonfinite")
		}
	}
	return nil
}
func (s WeightSampleInputs) build(g *transformerTapeGraph) (int, int, [2]int) {
	frames := len(s.DiagonalTime)
	// Scatter each column into [frames,2], then add without reassociation of MLP
	// arithmetic. Raw times are already admitted by the step; padding is finite.
	columns := func(ai, bi int) int {
		zero := g.zero(1)
		ac := g.row(ai, zero, frames, 1, 2, 0, 1)
		bc := g.row(bi, zero, frames, 1, 2, 1, 1)
		return g.sum(ac, bc, frames*2)
	}
	concat := func(a, b, size int) int {
		return g.sum(g.pad(a, 0, 2*size), g.pad(b, size, 2*size), 2*size)
	}
	di := g.constant([]int{frames}, s.DiagonalTime)
	si, ti := g.constant([]int{frames}, s.DistillS), g.constant([]int{frames}, s.DistillT)
	diag, dist := columns(di, di), columns(si, ti)
	input := g.reshape(concat(diag, dist, frames*2), 2*frames, 2)
	ds, ts := g.constant([]int{frames}, s.DiagonalSeed), g.constant([]int{frames}, s.DistillSeed)
	seed := g.reshape(concat(ds, ts, frames), 2*frames, 1)
	return input, seed, [2]int{ds, ts}
}

// BuildSampleWeightTrainingTape fuses sample/seed packing with the weighting
// MLP. Seed objective scaling has already occurred; no new scaling is applied.
// Full source admission and byte ceilings precede materialisation/pruning.
func (m *LSDWeightMLP) BuildSampleWeightTrainingTape(samples WeightSampleInputs, maxBytes int64) (WeightTrainingTape, error) {
	if len(samples.DiagonalTime) < 1 || len(samples.DiagonalTime) > 512 {
		return WeightTrainingTape{}, fmt.Errorf("sample weighting frame ceiling")
	}
	return m.buildWeightTrainingTape(nil, nil, 2*len(samples.DiagonalTime), maxBytes, &samples, false)
}

// TrainingSampleWeightBackend is optional and explicit. Legacy scopes continue
// to receive the host packed matrix via TrainingWeightBackend.
type TrainingSampleWeightBackend interface {
	SampleWeightEnabled() bool
	EvaluateSampleWeight(*LSDWeightMLP, WeightSampleInputs, bool) (WeightTrainingResult, error)
}

func (b *ResidentTrainingBackend) SampleWeightEnabled() bool {
	return b != nil && (b.ownsLifetime() || b.Scope == "ptx-resident-sampleweight-hybrid-v1" || b.Scope == "vulkan-resident-sampleweight-hybrid-v1")
}
func (b *ResidentTrainingBackend) EvaluateSampleWeight(m *LSDWeightMLP, samples WeightSampleInputs, accumulate bool) (WeightTrainingResult, error) {
	var empty WeightTrainingResult
	if err := b.admitGraph(); err != nil {
		return empty, err
	}
	if !b.SampleWeightEnabled() || b.trainer.Weighting != m || m.TrainingBackend != b {
		return empty, b.fail(fmt.Errorf("resident sample weighting owner/scope mismatch"))
	}
	if len(samples.DiagonalTime) < 1 || len(samples.DiagonalTime) > 512 {
		return empty, b.fail(fmt.Errorf("sample weighting frame ceiling"))
	}
	plan, err := m.buildWeightTrainingTape(nil, nil, 2*len(samples.DiagonalTime), b.MaxTapeBytes, &samples, true)
	if err != nil {
		return empty, b.fail(err)
	}
	requested := plan.Tape.Outputs[:2]
	if !accumulate && zeroTrainingSeed(samples.DiagonalSeed) && zeroTrainingSeed(samples.DistillSeed) {
		requested = plan.Tape.Outputs[:1]
	}
	out, err := b.runResidentTape(plan.Tape, plan.ResidentBindings, requested, accumulate, 1)
	if err != nil {
		return empty, b.fail(err)
	}
	if len(requested) == 1 {
		out = residentForwardBoundary(out, []int{2 * len(samples.DiagonalTime), 4 * len(samples.DiagonalTime)})
	}
	result := WeightTrainingResult{Output: out[0], DInput: out[1], Parameters: map[string][]float32{}}
	if err = validateWeightBoundary(result, 2*len(samples.DiagonalTime), samples.DiagonalTime, samples.DistillS, samples.DistillT, samples.DiagonalSeed, samples.DistillSeed); err != nil {
		return empty, b.fail(err)
	}
	return result, nil
}
