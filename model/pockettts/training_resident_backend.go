package pockettts

import (
	"context"
	"fmt"
	"strings"

	train "github.com/rcarmo/go-pherence/backends/training"
)

// ResidentTrainingBackend binds transformer and flow parameters/gradients and
// all optimiser state to one explicit owner. Extended scopes also own weighting,
// projections, embedding/BOS/sequence, time-embedding and EOS loss tapes. CPU
// retains dispatch/boundary assembly and diagnostic/snapshot transfers. Global
// norm scopes compute clipping on device; older scopes reduce summaries on host.
// Extended scopes also execute row branch and step gradient/metric tapes. Older scopes
// keep their host time embeddings, loss and interpolation arithmetic. Tape boundaries
// transfer through host storage. Full coherent host state synchronises each
// completed update; this is a resident-graph hybrid, not full-device training.
type ResidentTrainingBackend struct {
	Driver                         train.ResidentDriver
	Scope                          string
	MaxResidentBytes, MaxTapeBytes int64
	state                          *train.ResidentState
	host                           FullTrainingState
	sharedInitialMoments           []NamedAdamState          // private step-zero download storage
	forwardZeroWorkspace           *TrainingWorkspace        // borrowed only during one serialised step
	transformerStages              *transformerStageBoundary // private paired-stage owner
	stagedBackbone                 *stagedBackboneBoundary
	trainer                        *FullTrainer
	ctx                            context.Context
	active, closed                 bool
	terminal                       error
	handoffLease                   *train.ResidentTapeLease
	handoffHidden                  []float32
	handoffRoot, handoffRow        int
	handoffFrames, handoffWidth    int
	predictionLease                *train.ResidentTapeLease
	predictionValues               [][]float32
	predictionRoots                []int
	predictionKind                 string
	predictionRow                  int
	deviceCondition                *train.ResidentTapeLease
	deviceDiagonal                 *diagonalDeviceSources
	deviceDistill                  *distillDeviceSources
	deviceValues, deviceCopies     [][]float32
	deviceKind                     string
	deviceRow                      int
}

// TrainingStepPreparation admits residency before forward. Only the runner or
// an explicit deterministic step caller may open one paired step at a time.
type TrainingStepPreparation interface {
	PrepareTrainingStep(context.Context, *FullTrainer) error
}

func (b *ResidentTrainingBackend) TrainingOptimizerBackendName() string {
	if b == nil {
		return "unknown"
	}
	return b.Scope
}
func (b *ResidentTrainingBackend) TrainingFlowBackendName() string {
	return b.TrainingOptimizerBackendName()
}
func (b *ResidentTrainingBackend) TrainingTransformerBackendName() string {
	return b.TrainingOptimizerBackendName()
}
func (b *ResidentTrainingBackend) PrepareTrainingStep(ctx context.Context, t *FullTrainer) error {
	if b == nil || ctx == nil || t == nil || b.Driver == nil || b.closed || b.active || b.terminal != nil {
		return fmt.Errorf("nil/closed/active/terminal resident graph backend")
	}
	if b.Scope != "ptx-resident-hybrid-v1" && b.Scope != "vulkan-resident-hybrid-v1" && !b.ownsWeight() {
		return fmt.Errorf("unsupported resident graph scope")
	}
	if b.MaxResidentBytes <= 0 || b.MaxTapeBytes <= 0 {
		return fmt.Errorf("resident graph byte ceilings required")
	}
	if _, ok := b.Driver.(train.ResidentTapeDriver); !ok {
		return fmt.Errorf("resident graph driver lacks binding support")
	}
	if b.flowHandoffEnabled() {
		if _, ok := b.Driver.(train.ResidentHandoffTapeDriver); !ok {
			return b.fail(fmt.Errorf("resident driver lacks handoff preflight"))
		}
	}
	if b.embeddingCompactEnabled() {
		if _, ok := b.Driver.(train.ResidentCompactEmbeddingDriver); !ok {
			return fmt.Errorf("resident graph driver lacks compact embedding accumulation")
		}
	}
	if b.ownsZeroConstants() {
		if _, ok := b.Driver.(train.ResidentZeroTapeDriver); !ok {
			return fmt.Errorf("resident graph driver lacks zero-constant sharing")
		}
	}
	if b.ownsLifetime() {
		if _, ok := b.Driver.(train.ResidentReuseTapeDriver); !ok {
			return fmt.Errorf("resident graph driver lacks lifetime storage")
		}
	}
	if b.sharedWorkspaceEnabled() {
		if _, ok := b.Driver.(train.ResidentSharedWorkspaceDriver); !ok {
			return fmt.Errorf("resident graph driver lacks guarded sharedworkspace")
		}
		if _, ok := b.Driver.(train.ResidentDeviceScaleDriver); !ok {
			return fmt.Errorf("resident graph driver lacks shared device scale")
		}
		if _, ok := b.Driver.(train.ResidentGlobalNormDriver); !ok {
			return fmt.Errorf("resident graph driver lacks shared global norm")
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if t.OptimizerBackend != b || t.Flow.TrainingBackend != b || t.FlowLM.TransformerBackend != b || t.FlowLM.AttentionBackwardBackend != nil || !b.weightOwnerMatches(t) || !b.projectionOwnerMatches(t) || !b.sequenceOwnerMatches(t) || !b.eosOwnerMatches(t) || !b.lossOwnerMatches(t) || !b.branchOwnerMatches(t) || !b.stepReductionOwnerMatches(t) {
		return fmt.Errorf("resident graph components do not share owner")
	}
	if b.state == nil {
		var host FullTrainingState
		var err error
		if b.sharedWorkspaceEnabled() && b.host.Params != nil {
			if b.trainer != t || !sameResidentLiveBoundary(b.host, t) {
				return fmt.Errorf("shared restored host boundary changed")
			}
			host = b.host
		} else {
			host, err = t.State()
			if err != nil {
				return err
			}
		}
		var snapshot train.ResidentSnapshot
		if b.sharedWorkspaceEnabled() {
			snapshot, err = sharedInitialSnapshot(host)
		} else {
			snapshot, err = ResidentSnapshotFromFullState(host)
		}
		if err != nil {
			return err
		}
		if b.sharedWorkspaceEnabled() {
			// This scope owns every trainable gradient through bound tapes. Check
			// the complete inventory before omitting host-upload branch storage.
			for _, binding := range t.bindings {
				if !b.ownsResidentGradient(binding.name) {
					return fmt.Errorf("shared tape-only gradient owner missing for %s", binding.name)
				}
			}
			b.state, err = train.NewTapeOnlySharedWorkspaceResidentState(ctx, b.Driver, snapshot, b.MaxResidentBytes)
		} else if b.ownsStateCheck() {
			b.state, err = train.NewStateCheckResidentState(ctx, b.Driver, snapshot, b.MaxResidentBytes)
		} else if b.ownsDeviceScale() {
			b.state, err = train.NewDeviceScaleResidentState(ctx, b.Driver, snapshot, b.MaxResidentBytes)
		} else if b.ownsGlobalNorm() {
			b.state, err = train.NewGlobalNormResidentState(ctx, b.Driver, snapshot, b.MaxResidentBytes)
		} else {
			b.state, err = train.NewResidentState(ctx, b.Driver, snapshot, b.MaxResidentBytes)
		}
		if err != nil {
			if b.state != nil {
				b.terminal = err
			}
			return err
		}
		b.host = host
		if b.sharedWorkspaceEnabled() {
			b.rememberSharedInitialMoments(snapshot)
		}
	} else if b.trainer != t || !sameResidentLiveBoundary(b.host, t) {
		return fmt.Errorf("resident graph host boundary changed")
	}
	b.trainer, b.ctx, b.active = t, ctx, true
	return nil
}
func (b *ResidentTrainingBackend) admitGraph() error {
	if b == nil || b.closed || b.terminal != nil || !b.active || b.state == nil {
		return fmt.Errorf("resident graph requires prepared active step")
	}
	if b.trainer.OptimizerBackend != b || b.trainer.Flow.TrainingBackend != b || b.trainer.FlowLM.TransformerBackend != b || b.trainer.FlowLM.AttentionBackwardBackend != nil || !b.weightOwnerMatches(b.trainer) || !b.projectionOwnerMatches(b.trainer) || !b.sequenceOwnerMatches(b.trainer) || !b.eosOwnerMatches(b.trainer) || !b.lossOwnerMatches(b.trainer) || !b.branchOwnerMatches(b.trainer) || !b.stepReductionOwnerMatches(b.trainer) {
		return b.fail(fmt.Errorf("resident graph owner changed during step"))
	}
	return b.ctx.Err()
}
func (b *ResidentTrainingBackend) fail(err error) error {
	if err != nil {
		if b.terminal == nil {
			b.terminal = err
		}
		if b.handoffLease != nil {
			b.handoffLease.RetainFailure(err)
		}
	}
	return err
}
func (b *ResidentTrainingBackend) EvaluateFlow(*FlowHeadCPU, []float32, []float32, []float32, int, []float32, []float32) (FlowDualTrainingResult, error) {
	return FlowDualTrainingResult{}, fmt.Errorf("resident flow requires explicit accumulation policy")
}
func (b *ResidentTrainingBackend) EvaluateFlowAccumulating(m *FlowHeadCPU, c, t, x []float32, index int, gv, gt []float32, accumulate bool, scale float32) (FlowDualTrainingResult, error) {
	if err := b.admitGraph(); err != nil {
		return FlowDualTrainingResult{}, err
	}
	if m != b.trainer.Flow {
		return FlowDualTrainingResult{}, b.fail(fmt.Errorf("resident flow model owner changed"))
	}
	var tape FlowDualTrainingTape
	var err error
	tape, err = m.buildFlowDualTrainingTape(c, t, x, index, gv, gt, b.MaxTapeBytes, b.ownsTime(), true)
	if err != nil {
		return FlowDualTrainingResult{}, b.fail(err)
	}
	boundary := tape.boundarySize()
	outputs := tape.Tape.Outputs[:boundary]
	forward := b.ownsPruned() && !accumulate && zeroTrainingSeed(gv) && zeroTrainingSeed(gt)
	if forward {
		outputs = outputs[:2]
	}
	values, err := b.runResidentTape(tape.Tape, tape.ResidentBindings, outputs, accumulate, scale)
	if err != nil {
		return FlowDualTrainingResult{}, b.fail(err)
	}
	if forward {
		lengths := []int{len(gv), len(gv), len(c), len(x)}
		for range tape.timeEmbeddings {
			lengths = append(lengths, 1)
		}
		values = residentForwardBoundary(values, lengths)
	}
	result, err := tape.decodeBoundary(values)
	if err != nil {
		return FlowDualTrainingResult{}, b.fail(err)
	}
	return result, nil
}
func (b *ResidentTrainingBackend) EvaluateTransformer(*TransformerCPU, []float32, []float32, int) (TransformerTrainingResult, error) {
	return TransformerTrainingResult{}, fmt.Errorf("resident transformer requires explicit accumulation policy")
}
func (b *ResidentTrainingBackend) EvaluateTransformerAccumulating(m *TransformerCPU, x, seed []float32, rows int, accumulate bool) (TransformerTrainingResult, error) {
	if err := b.admitGraph(); err != nil {
		return TransformerTrainingResult{}, err
	}
	if m != b.trainer.FlowLM.Transformer {
		return TransformerTrainingResult{}, b.fail(fmt.Errorf("resident transformer model owner changed"))
	}
	tape, err := m.buildTransformerTrainingTape(x, seed, rows, b.MaxTapeBytes, true)
	if err != nil {
		return TransformerTrainingResult{}, b.fail(err)
	}
	outputs := tape.Tape.Outputs[:2]
	forward := b.ownsPruned() && !accumulate && zeroTrainingSeed(seed)
	if forward {
		outputs = outputs[:1]
	}
	values, err := b.runResidentTape(tape.Tape, tape.ResidentBindings, outputs, accumulate, 1)
	if err != nil {
		return TransformerTrainingResult{}, b.fail(err)
	}
	if forward {
		values = residentForwardBoundary(values, []int{len(x), len(x)})
	}
	return TransformerTrainingResult{Output: values[0], DSequence: values[1]}, nil
}
func (b *ResidentTrainingBackend) ownsWeight() bool {
	return b != nil && (b.Scope == "ptx-resident-weight-hybrid-v1" || b.Scope == "vulkan-resident-weight-hybrid-v1" || b.ownsProjection())
}
func (b *ResidentTrainingBackend) weightOwnerMatches(t *FullTrainer) bool {
	if b.ownsWeight() {
		return t.Weighting.TrainingBackend == b
	}
	return t.Weighting.TrainingBackend == nil
}
func (b *ResidentTrainingBackend) EvaluateWeight(m *LSDWeightMLP, input, seed []float32, rows int, accumulate bool) (WeightTrainingResult, error) {
	if err := b.admitGraph(); err != nil {
		return WeightTrainingResult{}, err
	}
	if !b.ownsWeight() || m != b.trainer.Weighting {
		return WeightTrainingResult{}, b.fail(fmt.Errorf("resident weighting owner/scope changed"))
	}
	tape, err := m.buildWeightTrainingTape(input, seed, rows, b.MaxTapeBytes, nil, true)
	if err != nil {
		return WeightTrainingResult{}, b.fail(err)
	}
	outputs := tape.Tape.Outputs[:2]
	forward := b.ownsPruned() && !accumulate && zeroTrainingSeed(seed)
	if forward {
		outputs = outputs[:1]
	}
	values, err := b.runResidentTape(tape.Tape, tape.ResidentBindings, outputs, accumulate, 1)
	if err != nil {
		return WeightTrainingResult{}, b.fail(err)
	}
	if forward {
		values = residentForwardBoundary(values, []int{len(seed), len(input)})
	}
	return WeightTrainingResult{Output: values[0], DInput: values[1]}, nil
}
func (b *ResidentTrainingBackend) ownsProjection() bool {
	return b != nil && (b.Scope == "ptx-resident-projection-hybrid-v1" || b.Scope == "vulkan-resident-projection-hybrid-v1" || b.ownsSequence())
}
func (b *ResidentTrainingBackend) projectionOwnerMatches(t *FullTrainer) bool {
	if b.ownsProjection() {
		return t.FlowLM.ProjectionBackend == b
	}
	return t.FlowLM.ProjectionBackend == nil
}
func projectionParameter(name string) bool {
	return name == "flow_lm.speaker_proj_weight" || name == "flow_lm.input_linear.weight" || name == "flow_lm.out_eos.weight" || name == "flow_lm.out_eos.bias"
}
func (b *ResidentTrainingBackend) EvaluateProjection(m *FlowLMTrainingCPU, kind string, input, seed []float32, rows int, accumulate bool) (ProjectionTrainingResult, error) {
	if err := b.admitGraph(); err != nil {
		return ProjectionTrainingResult{}, err
	}
	if !b.ownsProjection() || m != b.trainer.FlowLM {
		return ProjectionTrainingResult{}, b.fail(fmt.Errorf("resident projection owner/scope changed"))
	}
	var linear LinearF32
	var weight, bias string
	switch kind {
	case "speaker-empty":
		if rows != 0 || len(input) != 0 || len(seed) != 0 || !accumulate {
			return ProjectionTrainingResult{}, b.fail(fmt.Errorf("invalid empty speaker gradient"))
		}
		// One zero row produces an explicit zero weight gradient without a
		// host allocation/upload proportional to the full projection weight.
		return b.EvaluateProjection(m, "speaker", make([]float32, m.LatentDim), make([]float32, m.Hidden), 1, true)
	case "speaker":
		linear, weight = m.SpeakerProjection, "flow_lm.speaker_proj_weight"
	case "input":
		linear, weight = m.Input, "flow_lm.input_linear.weight"
	case "eos":
		linear, weight, bias = m.EOS, "flow_lm.out_eos.weight", "flow_lm.out_eos.bias"
	default:
		return ProjectionTrainingResult{}, b.fail(fmt.Errorf("unknown resident projection"))
	}
	tape, err := buildProjectionTrainingTape(linear, weight, bias, input, seed, rows, b.MaxTapeBytes, true)
	if err != nil {
		return ProjectionTrainingResult{}, b.fail(err)
	}
	outputs := tape.Tape.Outputs[:2]
	forward := b.ownsPruned() && !accumulate && zeroTrainingSeed(seed)
	if forward {
		outputs = outputs[:1]
	}
	values, err := b.runResidentTape(tape.Tape, tape.ResidentBindings, outputs, accumulate, 1)
	if err != nil {
		return ProjectionTrainingResult{}, b.fail(err)
	}
	if forward {
		values = residentForwardBoundary(values, []int{len(seed), len(input)})
	}
	return ProjectionTrainingResult{Output: values[0], DInput: values[1]}, nil
}
func residentGraphParameter(name string) bool {
	return strings.HasPrefix(name, "flow_lm.flow_net.") || strings.HasPrefix(name, "flow_lm.transformer.layers.") || strings.HasPrefix(name, "flow_lm.out_norm.")
}
func (b *ResidentTrainingBackend) ownsResidentGradient(name string) bool {
	return residentGraphParameter(name) || (b.ownsWeight() && strings.HasPrefix(name, "flow.w_s_t.")) || (b.ownsProjection() && projectionParameter(name)) || (b.ownsSequence() && sequenceParameter(name))
}

func (b *ResidentTrainingBackend) UpdateTrainer(ctx context.Context, t *FullTrainer, g TrainingStepGradients, maxNorm float32) error {
	if err := b.admitGraph(); err != nil {
		return err
	}
	if b.transformerStages != nil || b.stagedBackbone != nil {
		return b.fail(fmt.Errorf("transformer stage pair incomplete"))
	}
	if ctx == nil || t != b.trainer {
		return b.fail(fmt.Errorf("resident update context/trainer changed"))
	}
	if err := ctx.Err(); err != nil {
		return b.fail(err)
	}
	if !isFinite(maxNorm) || maxNorm <= 0 {
		return b.fail(fmt.Errorf("invalid resident norm ceiling"))
	}
	if !sameResidentLiveBoundary(b.host, t) {
		return b.fail(fmt.Errorf("host mutated during resident graph step"))
	}
	if !flowLMGradientMatches(g.FlowLM, t.FlowLM) || g.Flow == nil || len(g.Flow.Time) != len(t.Flow.Time) || len(g.Flow.Blocks) != len(t.Flow.Blocks) || g.Weighting == nil || len(g.Weighting.Layers) != len(t.Weighting.Layers) {
		return b.fail(fmt.Errorf("resident graph gradient topology"))
	}
	branches := map[string][]float32{}
	for _, binding := range t.bindings {
		values := binding.gradient(g)
		if len(values) != len(binding.parameter) || !finiteF32(values) {
			return b.fail(fmt.Errorf("resident graph gradient %s shape/nonfinite", binding.name))
		}
		if b.ownsResidentGradient(binding.name) {
			for _, v := range values {
				if v != 0 {
					return b.fail(fmt.Errorf("device-owned gradient %s also returned on host", binding.name))
				}
			}
		} else {
			branches[binding.name] = values
		}
	}
	if len(branches) > 0 {
		if err := b.state.Accumulate(ctx, branches, 1); err != nil {
			return b.fail(err)
		}
	}
	if _, err := b.state.Update(ctx, maxNorm); err != nil {
		return b.fail(err)
	}
	if b.sharedWorkspaceEnabled() {
		if err := b.loadSharedCompletedSnapshot(ctx, t); err != nil {
			return b.fail(err)
		}
	} else {
		snapshot, err := b.state.Snapshot(ctx)
		if err != nil {
			return b.fail(err)
		}
		coherent, err := takeFullStateFromResidentSnapshot(b.host, &snapshot)
		if err != nil {
			return b.fail(err)
		}
		if err = t.loadResidentCompletedState(coherent); err != nil {
			return b.fail(err)
		}
		b.host = coherent
	}
	b.active = false
	return nil
}
func (b *ResidentTrainingBackend) ValidateBoundary(t *FullTrainer) error {
	if b == nil || b.closed || b.active || b.transformerStages != nil || b.stagedBackbone != nil || b.terminal != nil || t == nil {
		return fmt.Errorf("resident graph boundary unavailable")
	}
	if b.state == nil {
		if b.sharedWorkspaceEnabled() && b.host.Params != nil && (b.trainer != t || !sameResidentLiveBoundary(b.host, t)) {
			return fmt.Errorf("shared restored checkpoint boundary mismatch")
		}
		return nil
	}
	if b.trainer != t || b.state.StepCount() != uint64(t.StepCount) || !sameResidentLiveBoundary(b.host, t) {
		return fmt.Errorf("resident graph checkpoint boundary mismatch")
	}
	return nil
}
func (b *ResidentTrainingBackend) Close() error {
	if b == nil || b.closed {
		return nil
	}
	if b.active || b.terminal != nil {
		return fmt.Errorf("resident graph resources retained at active/failed boundary: %v", b.terminal)
	}
	if err := b.state.Close(); err != nil {
		return b.fail(err)
	}
	if driver, ok := b.Driver.(interface{ Close() error }); ok {
		if err := driver.Close(); err != nil {
			return b.fail(err)
		}
	}
	b.closed = true
	return nil
}
