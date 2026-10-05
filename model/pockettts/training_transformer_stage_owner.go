package pockettts

import (
	"fmt"
	"slices"
)

// Private paired transformer stage owner. Boundary storage never escapes; no
// caller-supplied ticket, mutable parameter update or retry crosses the pair.
// The large-sequence shared runner selects these paired, bounded sources.
type transformerStageBoundary struct {
	owner      *ResidentTrainingBackend
	model      *TransformerCPU
	rows       int
	boundaries [][]float32
	consumed   bool
}

func (b *ResidentTrainingBackend) beginTransformerStages(m *TransformerCPU, sequence []float32, rows int) ([]float32, error) {
	if err := b.admitGraph(); err != nil {
		return nil, err
	}
	if m != b.trainer.FlowLM.Transformer || b.transformerStages != nil || b.handoffLease != nil || !sameResidentLiveBoundary(b.host, b.trainer) {
		return nil, b.fail(fmt.Errorf("transformer stage owner/active lease"))
	}
	seed := make([]float32, len(sequence))
	if err := validateTransformerStageModel(m, sequence, seed, rows); err != nil {
		return nil, b.fail(err)
	}
	var pool *trainingZeroPool
	if ws := b.forwardZeroWorkspace; ws != nil {
		if ws.zeroGradientOwner != b || ws.zeroGradientPool == nil || !ws.compatible(b.trainer.FlowLM, b.trainer.Flow) {
			return nil, b.fail(fmt.Errorf("transformer stage zero workspace provenance"))
		}
		pool = ws.zeroGradientPool
	}
	// Preflight every declared source before the first execution. Graphs are
	// rebuilt sequentially below to avoid retaining all native/host stage plans.
	for i := range m.Layers {
		stage, err := buildTransformerLayerStage(m, i, sequence, seed, rows, true, b.MaxTapeBytes, pool)
		if err != nil {
			return nil, b.fail(err)
		}
		if err = b.admitStagedStorage(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[:1], false); err != nil {
			return nil, b.fail(err)
		}
	}
	if m.FinalWeight != nil {
		stage, err := buildTransformerFinalNormStage(m, sequence, seed, rows, true, b.MaxTapeBytes, pool)
		if err != nil {
			return nil, b.fail(err)
		}
		if err = b.admitStagedStorage(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[:1], false); err != nil {
			return nil, b.fail(err)
		}
	}
	ticket := &transformerStageBoundary{owner: b, model: m, rows: rows, boundaries: [][]float32{append([]float32(nil), sequence...)}}
	b.transformerStages = ticket
	for i := range m.Layers {
		stage, err := buildTransformerLayerStage(m, i, ticket.boundaries[i], seed, rows, true, b.MaxTapeBytes, pool)
		if err != nil {
			return nil, b.fail(err)
		}
		if err = b.admitStagedStorage(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[:1], false); err != nil {
			return nil, b.fail(err)
		}
		values, err := b.runResidentTape(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[:1], false, 1)
		if err != nil {
			return nil, b.fail(err)
		}
		if len(values) != 1 || len(values[0]) != len(sequence) || !finiteF32(values[0]) {
			return nil, b.fail(fmt.Errorf("transformer stage forward boundary"))
		}
		ticket.boundaries = append(ticket.boundaries, values[0])
	}
	output := ticket.boundaries[len(ticket.boundaries)-1]
	if m.FinalWeight != nil {
		stage, err := buildTransformerFinalNormStage(m, output, seed, rows, true, b.MaxTapeBytes, pool)
		if err != nil {
			return nil, b.fail(err)
		}
		if err = b.admitStagedStorage(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[:1], false); err != nil {
			return nil, b.fail(err)
		}
		values, err := b.runResidentTape(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[:1], false, 1)
		if err != nil {
			return nil, b.fail(err)
		}
		if len(values) != 1 || len(values[0]) != len(sequence) || !finiteF32(values[0]) {
			return nil, b.fail(fmt.Errorf("transformer final forward boundary"))
		}
		output = values[0]
	}
	return append([]float32(nil), output...), nil
}

func (b *ResidentTrainingBackend) backwardTransformerStages(m *TransformerCPU, sequence, seed []float32, rows int) ([]float32, error) {
	if err := b.admitGraph(); err != nil {
		return nil, err
	}
	ticket := b.transformerStages
	if ticket == nil || ticket.owner != b || ticket.model != m || m != b.trainer.FlowLM.Transformer || ticket.rows != rows || ticket.consumed || !slices.Equal(sequence, ticket.boundaries[0]) {
		return nil, b.fail(fmt.Errorf("transformer stage backward provenance"))
	}
	if !sameResidentLiveBoundary(b.host, b.trainer) {
		return nil, b.fail(fmt.Errorf("transformer stage parameter boundary changed"))
	}
	if err := validateTransformerStageModel(m, sequence, seed, rows); err != nil {
		return nil, b.fail(err)
	}
	// All reverse sources are admitted before any parameter accumulation.
	for i := range m.Layers {
		stage, err := buildTransformerLayerStage(m, i, ticket.boundaries[i], seed, rows, false, b.MaxTapeBytes)
		if err != nil {
			return nil, b.fail(err)
		}
		if err = b.admitStagedStorage(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[1:2], true); err != nil {
			return nil, b.fail(err)
		}
	}
	if m.FinalWeight != nil {
		stage, err := buildTransformerFinalNormStage(m, ticket.boundaries[len(ticket.boundaries)-1], seed, rows, false, b.MaxTapeBytes)
		if err != nil {
			return nil, b.fail(err)
		}
		if err = b.admitStagedStorage(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[1:2], true); err != nil {
			return nil, b.fail(err)
		}
	}
	ticket.consumed = true
	gradient := append([]float32(nil), seed...)
	run := func(stage TransformerTrainingTape) error {
		// Re-admit the actual propagated seed: positive-zero sharing depends on
		// values, so the earlier original-seed plan is not an execution bound.
		if err := b.admitStagedStorage(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[1:2], true); err != nil {
			return err
		}
		// Separate boundary from gradient roots keeps the design payload bounded.
		values, err := b.runResidentTape(stage.Tape, stage.ResidentBindings, stage.Tape.Outputs[1:2], false, 1)
		if err != nil {
			return err
		}
		if len(values) != 1 || len(values[0]) != len(sequence) || !finiteF32(values[0]) {
			return fmt.Errorf("transformer stage reverse boundary")
		}
		if _, err = b.runGroupedStepBackbone(stage.Tape, stage.ResidentBindings, nil); err != nil {
			return err
		}
		gradient = values[0]
		return nil
	}
	if m.FinalWeight != nil {
		stage, err := buildTransformerFinalNormStage(m, ticket.boundaries[len(ticket.boundaries)-1], gradient, rows, false, b.MaxTapeBytes)
		if err != nil {
			return nil, b.fail(err)
		}
		if err = run(stage); err != nil {
			return nil, b.fail(err)
		}
	}
	for i := len(m.Layers) - 1; i >= 0; i-- {
		stage, err := buildTransformerLayerStage(m, i, ticket.boundaries[i], gradient, rows, false, b.MaxTapeBytes)
		if err != nil {
			return nil, b.fail(err)
		}
		if err = run(stage); err != nil {
			return nil, b.fail(err)
		}
	}
	b.transformerStages = nil
	return gradient, nil
}
