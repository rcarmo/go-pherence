package pockettts

import train "github.com/rcarmo/go-pherence/backends/training"

// Pruned scopes retain complete source admission/parameter inventory, but execute
// only declared boundary roots and the gradients actually accumulated. They do
// not reassociate arithmetic, fold zeros or reuse buffers across tape lifetimes.
func (b *ResidentTrainingBackend) ownsZeroConstants() bool {
	return b != nil && (b.EOSProjectionEnabled() || b.Scope == "ptx-resident-zeroconst-hybrid-v1" || b.Scope == "vulkan-resident-zeroconst-hybrid-v1")
}
func (b *ResidentTrainingBackend) ownsStateCheck() bool {
	return b != nil && (b.ownsZeroConstants() || b.Scope == "ptx-resident-statecheck-hybrid-v1" || b.Scope == "vulkan-resident-statecheck-hybrid-v1")
}
func (b *ResidentTrainingBackend) ownsLifetime() bool {
	return b != nil && (b.ownsStateCheck() || b.Scope == "ptx-resident-lifetime-hybrid-v1" || b.Scope == "vulkan-resident-lifetime-hybrid-v1")
}
func (b *ResidentTrainingBackend) ownsDeviceScale() bool {
	return b != nil && (b.SampleWeightEnabled() || b.Scope == "ptx-resident-devicescale-hybrid-v1" || b.Scope == "vulkan-resident-devicescale-hybrid-v1")
}
func (b *ResidentTrainingBackend) ownsGlobalNorm() bool {
	return b != nil && (b.ownsDeviceScale() || b.Scope == "ptx-resident-globalnorm-hybrid-v1" || b.Scope == "vulkan-resident-globalnorm-hybrid-v1")
}
func (b *ResidentTrainingBackend) ownsPruned() bool {
	return b != nil && (b.Scope == "ptx-resident-pruned-hybrid-v1" || b.Scope == "vulkan-resident-pruned-hybrid-v1" || b.ownsGlobalNorm())
}
func (b *ResidentTrainingBackend) runResidentTape(t train.Tape, bindings map[string]train.ResidentTapeBinding, outputs []int, accumulate bool, scale float32) ([][]float32, error) {
	if b.embeddingCompactEnabled() {
		var compact map[string]train.ResidentCompactEmbeddingBinding
		var err error
		t, bindings, compact, err = compactResidentEmbeddingTape(t, bindings, b.MaxTapeBytes)
		if err != nil {
			return nil, err
		}
		if b.handoffLease != nil {
			return b.handoffLease.RunCompactTape(b.ctx, t, bindings, compact, outputs, nil, accumulate, scale)
		}
		if compact != nil {
			return b.state.RunCompactZeroPrunedTape(b.ctx, t, bindings, compact, outputs, accumulate, scale, b.MaxTapeBytes)
		}
	}
	if b.embeddingIndexEnabled() {
		var err error
		t, err = indexResidentEmbeddingTape(t, bindings, b.MaxTapeBytes)
		if err != nil {
			return nil, err
		}
	}
	if b.ownsZeroConstants() {
		return b.state.RunZeroPrunedTape(b.ctx, t, bindings, outputs, accumulate, scale, b.MaxTapeBytes)
	}
	if b.ownsLifetime() {
		return b.state.RunReusePrunedTape(b.ctx, t, bindings, outputs, accumulate, scale, b.MaxTapeBytes)
	}
	if b.ownsPruned() {
		return b.state.RunPrunedTape(b.ctx, t, bindings, outputs, accumulate, scale, b.MaxTapeBytes)
	}
	return b.state.RunTape(b.ctx, t, bindings, outputs, accumulate, scale, b.MaxTapeBytes)
}
func (b *ResidentTrainingBackend) runUtilityTape(t train.Tape, outputs []int) ([][]float32, error) {
	if b.handoffLease != nil {
		return b.handoffLease.RunTape(b.ctx, t, nil, outputs, nil, false, 1)
	}
	if b.ownsZeroConstants() {
		return b.state.RunZeroPrunedUnboundTape(b.ctx, t, outputs, b.MaxTapeBytes)
	}
	if b.ownsLifetime() {
		return b.state.RunReusePrunedUnboundTape(b.ctx, t, outputs, b.MaxTapeBytes)
	}
	if b.ownsPruned() {
		return b.state.RunPrunedUnboundTape(b.ctx, t, outputs, b.MaxTapeBytes)
	}
	return b.state.RunUnboundTape(b.ctx, t, outputs, b.MaxTapeBytes)
}
func zeroTrainingSeed(x []float32) bool {
	for _, v := range x {
		if v != 0 {
			return false
		}
	}
	return true
}

// Pure forward callers require owned zero derivative placeholders, not device
// zero-seed reverse execution/downloads. Construction still validates seeds.
func residentForwardBoundary(values [][]float32, lengths []int) [][]float32 {
	out := make([][]float32, len(lengths))
	for i, n := range lengths {
		if i < len(values) {
			out[i] = values[i]
		} else {
			out[i] = make([]float32, n)
		}
	}
	return out
}
