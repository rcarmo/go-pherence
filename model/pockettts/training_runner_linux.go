//go:build linux

package pockettts

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"math"
	"os"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// PreparedTrainingRow contains already-normalised target latents, raw voice
// latents and token IDs. Codec/word alignment/corpus preparation is separate.
// ValidFrames excludes existing padding; the runner never appends an EOS row.
type PreparedTrainingRow struct {
	Batch       FlowLMTrainingBatch `json:"batch"`
	ValidFrames int                 `json:"valid_frames"`
}

// TrainingRunner is an experimental single-owner training loop. SIMD is the
// default; explicit attention-backward or flow hybrids retain other work on CPU.
// It is not concurrent-safe. A failed/uncertain update poisons this runner;
// create a new one and restore a coherent bundle rather than retrying it.
// Sampling distribution, numerical bounds and voice quality are unqualified.
type TrainingRunner struct {
	trainer   *FullTrainer
	run       *TrainingRunState
	rows      []PreparedTrainingRow
	limits    TrainingRandomLimits
	recipe    TrainingRandomRecipe
	maxNorm   float32
	poison    error
	workspace *TrainingWorkspace
	samples   TrainingStepSamples
}

func NewTrainingRunner(trainer *FullTrainer, identity TrainingRunIdentity, rows []PreparedTrainingRow, seeds [6]uint64, limits TrainingRandomLimits, maxNorm float32, allowExperimental bool) (*TrainingRunner, error) {
	if !allowExperimental {
		return nil, fmt.Errorf("random-sampler training requires explicit experimental opt-in; numerics/distributions/quality remain unqualified")
	}
	if trainer == nil || trainer.FlowLM == nil || trainer.Flow == nil || trainer.Weighting == nil || trainer.StepCount != 0 || len(rows) == 0 || len(rows) > MaxTrainingRunRows || !isFinite(maxNorm) || maxNorm <= 0 || limits.Samples.MaxFrames <= 0 || limits.Samples.MaxNoiseElements <= 0 || limits.MaxRawDraws <= 0 {
		return nil, fmt.Errorf("invalid training runner/trainer/rows/gradient ceiling")
	}
	if trainingExecutionBackendName(trainer) != identity.Backend {
		return nil, fmt.Errorf("training backend does not match replay identity")
	}
	if err := trainer.Config.validate(); err != nil {
		return nil, err
	}
	if err := trainer.refreshParameterBindings(); err != nil {
		return nil, err
	}
	run, err := NewTrainingRunState(identity, len(rows), seeds)
	if err != nil {
		return nil, err
	}
	r := &TrainingRunner{trainer: trainer, run: run, limits: limits, maxNorm: maxNorm, recipe: TrainingRandomRecipe{Algorithm: TrainingRandomAlgorithm, OptionsSHA256: identity.OptionsSHA256, DistillProbability: 1, FlowBatchMultiplier: 1, StopGradient: "minimal", Normalize: true, PEqual: .75, EOSLossWeight: .1, LogitMean: .4, LogitStd: 1}}
	if err := r.validateUpdated(); err != nil {
		return nil, err
	}
	// Admission consumes no proposal draws. Reuse one isolated proposal rather
	// than cloning the complete permutation once per corpus row (quadratic).
	p, err := run.PrepareRow()
	if err != nil {
		return nil, err
	}
	if err := validateRandomProposal(p, identity.OptionsSHA256); err != nil {
		return nil, err
	}
	r.rows = make([]PreparedTrainingRow, len(rows))
	for i, row := range rows {
		b := row.Batch
		if b.Frames <= 0 || b.Frames > limits.Samples.MaxFrames || trainer.FlowLM.LatentDim < 2 || b.Frames > limits.Samples.MaxNoiseElements/trainer.FlowLM.LatentDim {
			return nil, fmt.Errorf("row %d exceeds runner sample ceilings", i)
		}
		// Validate caller input lengths/values before copying. The product below
		// is checked independently of the latent-element ceiling.
		hiddenElements, ok := checked.MulInt(b.Frames, trainer.FlowLM.Hidden)
		if !ok {
			return nil, fmt.Errorf("row %d hidden shape overflows", i)
		}
		if _, _, err := trainer.FlowLM.validate(b, make([]float32, hiddenElements), make([]float32, b.Frames)); err != nil {
			return nil, fmt.Errorf("row %d: %w", i, err)
		}
		// Deep-copy inputs: caller mutation must not change replayed training.
		b.NormalizedLatents = append([]float32(nil), b.NormalizedLatents...)
		b.VoiceLatents = append([]float32(nil), b.VoiceLatents...)
		b.TextTokens = append([]uint32(nil), b.TextTokens...)
		// Buffers are created disjointly by this runner. Check row-dependent
		// draw bounds without repeating O(corpus rows) proposal validation.
		if row.ValidFrames <= 0 || row.ValidFrames > b.Frames {
			return nil, fmt.Errorf("row %d valid-prefix mismatch", i)
		}
		full, ok := checked.MulInt(b.Frames, trainer.FlowLM.LatentDim)
		if !ok {
			return nil, fmt.Errorf("row %d latent shape overflows", i)
		}
		if _, ok := checked.MulInt(full, 4); !ok {
			return nil, fmt.Errorf("row %d latent bytes overflow", i)
		}
		noise, ok := checked.MulInt(row.ValidFrames, trainer.FlowLM.LatentDim)
		if !ok {
			return nil, fmt.Errorf("row %d noise shape overflow", i)
		}
		noiseWords, ok := checked.MulInt(noise, 2)
		if !ok {
			return nil, fmt.Errorf("row %d noise draw overflow", i)
		}
		timeWords, ok := checked.MulInt(row.ValidFrames, 6)
		if !ok {
			return nil, fmt.Errorf("row %d time draw overflow", i)
		}
		total, ok := checked.AddInt(noiseWords, timeWords)
		if !ok || total > limits.MaxRawDraws {
			return nil, fmt.Errorf("row %d raw draw ceiling", i)
		}
		r.rows[i] = PreparedTrainingRow{Batch: b, ValidFrames: row.ValidFrames}
	}
	return r, nil
}

func runnerSampleBuffers(frames, channels int) TrainingStepSamples {
	return TrainingStepSamples{Mask: make([]bool, frames), Noise: make([]float32, frames*channels), DiagonalTime: make([]float32, frames), DistillS: make([]float32, frames), DistillT: make([]float32, frames)}
}

func (r *TrainingRunner) Progress() (step, epoch uint64, cursor int) {
	if r == nil {
		return 0, 0, 0
	}
	return r.run.Progress()
}

// Step checks cancellation before starting an update. Mid-update cancellation
// is observed only at the next completed boundary; it cannot undo Adam/EMA.
func (r *TrainingRunner) Step(ctx context.Context) (TrainingStepMetrics, error) {
	var metrics TrainingStepMetrics
	if r == nil || ctx == nil {
		return metrics, fmt.Errorf("nil runner/context")
	}
	if r.poison != nil {
		return metrics, fmt.Errorf("training runner requires checkpoint recovery: %w", r.poison)
	}
	if trainingExecutionBackendName(r.trainer) != r.run.core.identity.Backend {
		return metrics, fmt.Errorf("training backend changed against replay identity")
	}
	if err := ctx.Err(); err != nil {
		return metrics, err
	}
	step, _, _ := r.run.Progress()
	if step >= uint64(math.MaxInt) || r.trainer.StepCount != int(step) {
		return metrics, fmt.Errorf("trainer/run step mismatch or overflow")
	}
	p, err := r.run.PrepareRow()
	if err != nil {
		return metrics, err
	}
	index, err := p.Row()
	if err != nil {
		return metrics, err
	}
	row := r.rows[index]
	// Keep only one current-shape workspace. Fixed-shape rows reuse gradient/
	// tape and sample storage; mixed shapes replace it rather than building an
	// unbounded cache of model-sized workspaces.
	if r.workspace == nil || r.workspace.frames != row.Batch.Frames {
		pooled := false
		if owner, ok := r.trainer.OptimizerBackend.(*ResidentTrainingBackend); ok && sharedZeroGradientOwner(owner, r.trainer.FlowLM, r.trainer.Flow) {
			if r.trainer.Weighting.TrainingBackend != owner {
				return metrics, fmt.Errorf("shared zero-gradient weighting owner changed")
			}
			pooled = true
		}
		workspace, err := newTrainingWorkspace(r.trainer.FlowLM, r.trainer.Flow, row.Batch, pooled)
		if err != nil {
			return metrics, err
		}
		r.workspace = workspace
		r.samples = runnerSampleBuffers(row.Batch.Frames, r.trainer.FlowLM.LatentDim)
	}
	if err := SampleTrainingInto(r.samples, p, row.Batch.Frames, row.ValidFrames, r.trainer.FlowLM.LatentDim, r.recipe, r.limits); err != nil {
		return metrics, err
	}
	if backend, ok := r.trainer.OptimizerBackend.(TrainingStepPreparation); ok {
		if err := backend.PrepareTrainingStep(ctx, r.trainer); err != nil {
			r.poison = err
			return metrics, err
		}
	}
	metrics, gradients, err := PocketTrainingStepInto(r.trainer.FlowLM, r.trainer.Flow, r.trainer.Weighting, row.Batch, r.samples, DefaultTrainingStepConfig(), r.workspace)
	if err != nil {
		// A device call may have submitted uncertain work even though host
		// parameters remain unchanged. Do not retry or checkpoint this owner.
		if r.run.core.identity.Backend != "simd" {
			r.poison = err
		}
		return metrics, err
	}
	if !finite64(metrics.Loss) {
		return metrics, fmt.Errorf("nonfinite training loss")
	}
	if r.trainer.OptimizerBackend != nil {
		// Device optimiser owns global clipping; never pre-clip on CPU too.
		if err := r.trainer.OptimizerBackend.UpdateTrainer(ctx, r.trainer, gradients, r.maxNorm); err != nil {
			r.poison = err
			return metrics, err
		}
	} else {
		if err := r.clipGradients(gradients); err != nil {
			return metrics, err
		}
		// Everything below can have changed live parameters; poison on error.
		if err := r.trainer.Step(gradients); err != nil {
			r.poison = err
			return metrics, err
		}
	}
	if err := r.validateUpdated(); err != nil {
		r.poison = err
		return metrics, err
	}
	if err := r.run.CommitRow(p, uint64(r.trainer.StepCount)); err != nil {
		r.poison = err
		return metrics, err
	}
	return metrics, nil
}
func (r *TrainingRunner) clipGradients(g TrainingStepGradients) error {
	// Finite F32 squares fit in F64 even summed across an int-sized slice.
	// Compensated accumulation avoids one transcendental Hypot call per
	// parameter; take a single square root after the validated reduction.
	var square, correction float64
	for _, b := range r.trainer.bindings {
		values := b.gradient(g)
		if len(values) != len(b.parameter) {
			return fmt.Errorf("gradient shape changed for %s", b.name)
		}
		for _, x := range values {
			if !isFinite(x) {
				return fmt.Errorf("nonfinite gradient %s", b.name)
			}
			term := float64(x)*float64(x) - correction
			next := square + term
			correction = (next - square) - term
			square = next
		}
	}
	norm := math.Sqrt(square)
	if !finite64(norm) {
		return fmt.Errorf("gradient norm overflow")
	}
	if norm > float64(r.maxNorm) {
		scale := float32(float64(r.maxNorm) / norm)
		for _, b := range r.trainer.bindings {
			values := b.gradient(g)
			for i := range values {
				values[i] *= scale
			}
		}
	}
	return nil
}
func (r *TrainingRunner) validateUpdated() error {
	for _, b := range r.trainer.bindings {
		for _, x := range b.parameter {
			if !isFinite(x) {
				return fmt.Errorf("updated parameter %s nonfinite", b.name)
			}
		}
		m, exists := r.trainer.Moments[b.name]
		if exists && (len(m.M) != len(b.parameter) || len(m.V) != len(b.parameter)) {
			return fmt.Errorf("updated moments %s shape mismatch", b.name)
		}
		if r.trainer.StepCount > 0 && !exists {
			return fmt.Errorf("updated moments %s missing", b.name)
		}
		for i, x := range m.M {
			if !isFinite(x) || !isFinite(m.V[i]) || m.V[i] < 0 {
				return fmt.Errorf("updated moments %s invalid", b.name)
			}
		}
		if r.trainer.EMADecay > 0 && len(r.trainer.EMA[b.name]) != len(b.parameter) {
			return fmt.Errorf("updated EMA %s shape mismatch", b.name)
		}
		for _, x := range r.trainer.EMA[b.name] {
			if !isFinite(x) {
				return fmt.Errorf("updated EMA %s nonfinite", b.name)
			}
		}
	}
	if !isFinite(r.trainer.EMADecay) || r.trainer.EMADecay < 0 || r.trainer.EMADecay >= 1 {
		return fmt.Errorf("invalid runner EMA decay")
	}
	return nil
}

// Checkpoint serialises decoder-valid state into a temporary regular file,
// then commits model+run metadata through the existing immutable bundle layer.
// This adds snapshot memory/file peaks; caller ceilings must include them.
// No old generation is deleted and publication-uncertain errors are preserved.
func (r *TrainingRunner) Checkpoint(directory, generation string, limits TrainingBundleLimits) (TrainingCheckpointBinding, error) {
	if r == nil || r.poison != nil {
		return TrainingCheckpointBinding{}, fmt.Errorf("cannot checkpoint nil/poisoned runner")
	}
	if trainingExecutionBackendName(r.trainer) != r.run.core.identity.Backend {
		return TrainingCheckpointBinding{}, fmt.Errorf("checkpoint backend identity changed")
	}
	if b := r.trainer.OptimizerBackend; b != nil {
		if err := b.ValidateBoundary(r.trainer); err != nil {
			return TrainingCheckpointBinding{}, err
		}
	}
	step, _, _ := r.run.Progress()
	if step != uint64(r.trainer.StepCount) {
		return TrainingCheckpointBinding{}, fmt.Errorf("checkpoint step mismatch")
	}
	if err := limits.validate(); err != nil {
		return TrainingCheckpointBinding{}, err
	}
	if !bundleGeneration(generation) {
		return TrainingCheckpointBinding{}, fmt.Errorf("invalid checkpoint generation")
	}
	var state FullTrainingState
	var err error
	if b, ok := r.trainer.OptimizerBackend.(*ResidentTrainingBackend); ok && b.sharedWorkspaceEnabled() && (b.state != nil || b.host.Params != nil) {
		state, err = b.sharedCheckpointState(r.trainer)
	} else {
		state, err = r.trainer.State()
	}
	if err != nil {
		return TrainingCheckpointBinding{}, err
	}
	file, err := os.CreateTemp(directory, ".runner-serialise-*.tmp")
	if err != nil {
		return TrainingCheckpointBinding{}, err
	}
	defer os.Remove(file.Name()) // Only this owned scratch file, never generations.
	defer file.Close()
	bounded := &runnerLimitWriter{writer: bufio.NewWriterSize(file, 64<<10), left: limits.MaxModelBytes}
	if limits.MaxModelBytes <= 0 {
		return TrainingCheckpointBinding{}, fmt.Errorf("checkpoint byte ceiling must be positive")
	}
	if err := writeFullTrainingBinary(bounded, state); err != nil {
		return TrainingCheckpointBinding{}, err
	}
	if err := bounded.writer.Flush(); err != nil {
		return TrainingCheckpointBinding{}, err
	}
	if _, err := file.Seek(0, 0); err != nil {
		return TrainingCheckpointBinding{}, err
	}
	return PublishTrainingBundle(directory, generation, file, step, r.run, limits)
}

type runnerLimitWriter struct {
	writer *bufio.Writer
	left   int64
}

func (w *runnerLimitWriter) Write(p []byte) (int, error) {
	if int64(len(p)) > w.left {
		return 0, fmt.Errorf("runner checkpoint byte ceiling exceeded")
	}
	n, err := w.writer.Write(p)
	w.left -= int64(n)
	return n, err
}

// Restore checks association and then the full binary state before mutating
// this fresh runner. An existing live/poisoned runner cannot be rolled back here.
func (r *TrainingRunner) Restore(directory string, limits TrainingBundleLimits) error {
	if r == nil || r.trainer == nil || r.run == nil || r.poison != nil || r.trainer.StepCount != 0 {
		return fmt.Errorf("restore requires a fresh runner")
	}
	if trainingExecutionBackendName(r.trainer) != r.run.core.identity.Backend {
		return fmt.Errorf("restore backend identity changed")
	}
	if b := r.trainer.OptimizerBackend; b != nil {
		if err := b.ValidateBoundary(r.trainer); err != nil {
			return err
		}
	}
	step, _, _ := r.run.Progress()
	if step != 0 {
		return fmt.Errorf("restore requires step-zero run")
	}
	bundle, err := OpenTrainingBundle(directory, r.run.core.identity, limits)
	if err != nil {
		return err
	}
	state, readErr := ReadFullTrainingStateBinary(bundle.Model, bundle.ModelBytes)
	closeErr := bundle.Model.Close()
	if err := errors.Join(readErr, closeErr); err != nil {
		return err
	}
	bundleStep, _, _ := bundle.Run.Progress()
	if bundle.Run.core.rows != len(r.rows) || uint64(state.Step) != bundleStep || state.AdamW != r.trainer.Config || state.EMADecay != r.trainer.EMADecay {
		return fmt.Errorf("checkpoint model/run/optimizer mismatch")
	}
	for _, moment := range state.Adam {
		for _, v := range moment.V {
			if v < 0 {
				return fmt.Errorf("checkpoint contains negative second moments")
			}
		}
	}
	if b, ok := r.trainer.OptimizerBackend.(*ResidentTrainingBackend); ok && b.sharedWorkspaceEnabled() && state.Step > 0 {
		if err := b.restoreSharedOwnedState(r.trainer, &state); err != nil {
			return err
		}
	} else if err := r.trainer.LoadState(state); err != nil {
		return err
	}
	r.run = bundle.Run
	return nil
}
