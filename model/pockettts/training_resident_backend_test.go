//go:build linux

package pockettts

import (
	"bytes"
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"

	train "github.com/rcarmo/go-pherence/backends/training"
)

type hostResidentGraphDriver struct {
	hostOptimizerDriver
	t                                                       *testing.T
	tapes, downloads, gradientDownloads                     int
	failTape                                                int
	accumulations                                           int
	validatedGradients                                      [][]float32
	graphCommands, graphTensors                             int
	graphBytes                                              int64
	storageBytes                                            int64
	legacyStorageBytes, ownedStorageBytes, pooledOwnedBytes int64
	afterRun                                                func()
	beforeRun                                               func(*hostResidentGraphTape)
	handoffAfterRun                                         func(*hostResidentGraphTape)
	handoffAfterClose                                       func(*hostResidentGraphTape)
	handoffBeforeClose                                      func(*hostResidentGraphTape) error
	afterDownload                                           func([][]float32)
	downloadCalls                                           int
	downloadedElements                                      int
	afterTapeClose                                          func() error
	afterAccumulate                                         func([]float32)
}
type hostResidentGraphTape struct {
	driver        *hostResidentGraphDriver
	tape          train.Tape
	values        map[int][]float32
	closed        bool
	reuse         bool
	zeroProtected map[int]bool
	borrowed      map[int][]float32
}

func (d *hostResidentGraphDriver) NewZeroBoundTape(ctx context.Context, tape train.Tape, parameters map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	protected := map[int]bool{}
	for id := range parameters {
		protected[id] = true
	}
	p, err := train.PlanZeroTapeStorage(tape, maxBytes, protected)
	if err != nil {
		return nil, err
	}
	d.storageBytes += p.Bytes
	owned := p.Bytes
	for id := range parameters {
		n, _ := tape.Tensors[id].Elements()
		owned -= int64(n) * 4
	}
	d.pooledOwnedBytes += owned
	owner, err := d.NewBoundTape(ctx, tape, parameters, maxBytes)
	if err != nil {
		return owner, err
	}
	out := owner.(*hostResidentGraphTape)
	out.reuse = true
	out.zeroProtected = protected
	return out, nil
}
func (d *hostResidentGraphDriver) NewReuseBoundTape(ctx context.Context, tape train.Tape, parameters map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	p, err := train.PlanTapeStorage(tape, maxBytes)
	if err != nil {
		return nil, err
	}
	d.storageBytes += p.Bytes
	owned := p.Bytes
	for id := range parameters {
		n, _ := tape.Tensors[id].Elements()
		owned -= int64(n) * 4
	}
	d.pooledOwnedBytes += owned
	owner, err := d.NewBoundTape(ctx, tape, parameters, maxBytes)
	if err != nil {
		return owner, err
	}
	owner.(*hostResidentGraphTape).reuse = true
	return owner, nil
}
func (d *hostResidentGraphDriver) NewBoundTape(ctx context.Context, tape train.Tape, parameters map[int]train.ResidentBuffer, maxBytes int64) (train.ResidentBoundTape, error) {
	aliases := map[int]bool{}
	for _, c := range tape.Commands {
		if c.Op == "reshape" {
			aliases[c.Output] = true
		}
	}
	for id, s := range tape.Tensors {
		if !aliases[id] {
			n, _ := s.Elements()
			d.legacyStorageBytes += int64(n) * 4
			if _, borrowed := parameters[id]; !borrowed {
				d.ownedStorageBytes += int64(n) * 4
			}
		}
	}
	d.tapes++
	d.graphCommands += len(tape.Commands)
	d.graphTensors += len(tape.Tensors)
	for _, s := range tape.Tensors {
		n, _ := s.Elements()
		d.graphBytes += int64(n) * 4
	}
	if d.failTape == d.tapes {
		return nil, errors.New("graph admission failed")
	}
	copyTape := tape
	copyTape.Tensors = append([]train.TensorSpec(nil), tape.Tensors...)
	var borrowed map[int][]float32
	if d.handoffAfterRun != nil {
		borrowed = map[int][]float32{}
	}
	for _, spec := range tape.Tensors {
		if spec.External {
			borrowed = map[int][]float32{}
			break
		}
	}
	for id, b := range parameters {
		if !copyTape.Tensors[id].External {
			copyTape.Tensors[id].Values = append([]float32(nil), b.(*hostOptimizerBuffer).values...)
		}
		if borrowed != nil {
			borrowed[id] = b.(*hostOptimizerBuffer).values
		}
	}
	return &hostResidentGraphTape{driver: d, tape: copyTape, borrowed: borrowed}, nil
}
func (d *hostResidentGraphDriver) ValidateAccumulate(context.Context, train.ResidentBuffer, train.ResidentBuffer, train.ResidentBuffer, int, float32) error {
	d.accumulations++
	return nil
}
func (d *hostResidentGraphDriver) Accumulate(ctx context.Context, out, previous, source train.ResidentBuffer, n int, scale float32) error {
	if err := d.hostOptimizerDriver.Accumulate(ctx, out, previous, source, n, scale); err != nil {
		return err
	}
	if d.afterAccumulate != nil {
		d.afterAccumulate(out.(*hostOptimizerBuffer).values)
	}
	return nil
}
func (d *hostResidentGraphDriver) ValidateUpdate(ctx context.Context, p, g, m, v, ema train.ResidentBuffer, n int, step train.AdamWStep) error {
	d.validatedGradients = append(d.validatedGradients, append([]float32(nil), g.(*hostOptimizerBuffer).values...))
	return d.hostOptimizerDriver.ValidateUpdate(ctx, p, g, m, v, ema, n, step)
}
func (t *hostResidentGraphTape) RunDevice(ctx context.Context) error {
	if t.driver.beforeRun != nil {
		t.driver.beforeRun(t)
	}
	var output [][]float32
	if t.zeroProtected != nil {
		output = interpretFlowTapeStorageBound(t.driver.t, t.tape, t.reuse, t.borrowed, t.zeroProtected)
	} else {
		output = interpretFlowTapeStorageBound(t.driver.t, t.tape, t.reuse, t.borrowed)
	}
	t.values = map[int][]float32{}
	for i, id := range t.tape.Outputs {
		t.values[id] = output[i]
	}
	if t.driver.afterRun != nil {
		t.driver.afterRun()
	}
	if t.driver.handoffAfterRun != nil {
		t.driver.handoffAfterRun(t)
	}
	return ctx.Err()
}
func (t *hostResidentGraphTape) OutputBuffer(id int) (train.ResidentBuffer, error) {
	x, ok := t.values[id]
	if !ok {
		return nil, errors.New("output id")
	}
	return &hostOptimizerBuffer{values: x, driver: &t.driver.hostOptimizerDriver}, nil
}
func (t *hostResidentGraphTape) DownloadOutputs(ctx context.Context, ids []int) ([][]float32, error) {
	t.driver.downloads += len(ids)
	t.driver.downloadCalls++
	values := make([][]float32, len(ids))
	for i, id := range ids {
		values[i] = append([]float32(nil), t.values[id]...)
		t.driver.downloadedElements += len(values[i])
	}
	if t.driver.afterDownload != nil {
		t.driver.afterDownload(values)
	}
	return values, ctx.Err()
}
func (t *hostResidentGraphTape) Close() error {
	if t.driver.handoffBeforeClose != nil {
		if err := t.driver.handoffBeforeClose(t); err != nil {
			return err
		}
	}
	if t.driver.afterTapeClose != nil {
		if err := t.driver.afterTapeClose(); err != nil {
			return err
		}
	}
	t.closed = true
	if t.driver.handoffAfterClose != nil {
		t.driver.handoffAfterClose(t)
	}
	return nil
}
func installResidentGraph(t *testing.T, trainer *FullTrainer, scope string) (*ResidentTrainingBackend, *hostResidentGraphDriver) {
	t.Helper()
	driver := &hostResidentGraphDriver{t: t}
	b := &ResidentTrainingBackend{Driver: driver, Scope: scope, MaxResidentBytes: 64 << 20, MaxTapeBytes: 64 << 20}
	trainer.OptimizerBackend = b
	trainer.Flow.TrainingBackend = b
	trainer.FlowLM.TransformerBackend = b
	if b.ownsWeight() {
		trainer.Weighting.TrainingBackend = b
	}
	if b.ownsProjection() {
		trainer.FlowLM.ProjectionBackend = b
	}
	if b.ownsSequence() {
		trainer.FlowLM.SequenceBackend = b
	}
	if b.ownsEOS() {
		trainer.FlowLM.EOSBackend = b
	}
	if b.ownsLoss() {
		trainer.Flow.LossBackend = b
	}
	if b.ownsBranch() {
		trainer.Flow.BranchBackend = b
	}
	if b.ownsStepReduction() {
		trainer.Flow.StepReductionBackend = b
	}
	return b, driver
}
func TestTrainingResidentGraphFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-hybrid-v1")
}
func TestTrainingResidentWeightFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-weight-hybrid-v1")
}
func TestTrainingResidentProjectionFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-projection-hybrid-v1")
}
func TestTrainingResidentSequenceFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-sequence-hybrid-v1")
}
func TestTrainingResidentTimeFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-time-hybrid-v1")
}
func TestTrainingResidentEOSFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-eos-hybrid-v1")
}
func TestTrainingResidentLossFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-loss-hybrid-v1")
}
func TestTrainingResidentBranchFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-branch-hybrid-v1")
}
func TestTrainingResidentStepFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-step-hybrid-v1")
}
func TestTrainingResidentPrunedFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-pruned-hybrid-v1")
}
func TestTrainingResidentGlobalNormFullStepPyTorch(t *testing.T) {
	testResidentGraphFullStep(t, "ptx-resident-globalnorm-hybrid-v1")
}
func testResidentGraphFullStep(t *testing.T, scope string) {
	testResidentGraphFullStepConfig(t, scope, DefaultTrainingStepConfig())
}
func testResidentGraphFullStepConfig(t *testing.T, scope string, config TrainingStepConfig) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	adam := DefaultAdamWConfig()
	a := oracle.AdamW
	adam.LearningRate, adam.Beta1, adam.Beta2, adam.Epsilon, adam.WeightDecay = a.LearningRate, a.Beta1, a.Beta2, a.Epsilon, a.WeightDecay
	trainer, err := NewFullTrainer(lm, flow, w, adam, a.EMADecay)
	if err != nil {
		t.Fatal(err)
	}
	backend, d := installResidentGraph(t, trainer, scope)
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	if err := backend.PrepareTrainingStep(context.Background(), trainer); err != nil {
		t.Fatal(err)
	}
	metrics, g, err := PocketTrainingStep(lm, flow, w, batch, samples, config)
	if err != nil {
		t.Fatal(err)
	}
	if config != DefaultTrainingStepConfig() {
		cpuLM, cpuFlow, cpuW := trainingStepModelsFromOracle(t, oracle)
		cpu, grad, err := PocketTrainingStep(cpuLM, cpuFlow, cpuW, batch, samples, config)
		if err != nil {
			t.Fatal(err)
		}
		oracle.Metrics["loss"] = cpu.Loss
		oracle.Gradients = trainingStepGradientMap(grad)
		oracle.DNormalizedLatents = grad.Inputs.NormalizedLatents
		oracle.DVoiceLatents = grad.Inputs.VoiceLatents
		reference, err := NewFullTrainer(cpuLM, cpuFlow, cpuW, adam, a.EMADecay)
		if err != nil {
			t.Fatal(err)
		}
		if err := reference.Step(grad); err != nil {
			t.Fatal(err)
		}
		oracle.AdamW.ParametersAfterStep = reference.params
		oracle.AdamW.EMAAfterStep = reference.EMA
	}
	assertClose64(t, "resident graph full loss", metrics.Loss, oracle.Metrics["loss"], 3e-5)
	if backend.ownsStepReduction() && config == DefaultTrainingStepConfig() {
		for name, got := range map[string]float64{"flow_diag": metrics.FlowDiagonal, "flow_distill": metrics.FlowDistill, "flow_loss": metrics.FlowLoss, "eos_loss": metrics.EOS} {
			want, ok := oracle.Metrics[name]
			if !ok {
				t.Fatal("metric fixture missing", name)
			}
			assertClose64(t, "step metric "+name, got, want, 3e-5)
		}
	}
	assertSliceClose(t, "resident graph target gradients", g.Inputs.NormalizedLatents, oracle.DNormalizedLatents, 5e-5)
	assertSliceClose(t, "resident graph voice gradients", g.Inputs.VoiceLatents, oracle.DVoiceLatents, 5e-5)
	// All 34 flow parameters accumulate twice per valid row; transformer once.
	// Forward and endpoint tapes contribute no parameter adjoints.
	flowParameters := map[string][]float32{}
	addFlowHeadParameters(flowParameters, flow)
	transformerParameters := map[string][]float32{}
	addTransformerParameters(transformerParameters, lm.Transformer)
	weightParameters := 0
	if backend.ownsWeight() {
		weightParameters = 2 * len(w.Layers)
	}
	projectionParameters := 0
	if backend.ownsProjection() {
		projectionParameters = 4
	}
	sequenceParameters := 0
	if backend.ownsSequence() {
		sequenceParameters = 3
	}
	if d.accumulations != 4*len(flowParameters)+len(transformerParameters)+weightParameters+projectionParameters+sequenceParameters {
		t.Fatal("endpoint/forward parameter gradient leaked", d.accumulations)
	}
	// Host gradient arrays for device-owned components are intentionally zero.
	for name, x := range trainingStepGradientMap(g) {
		if residentGraphParameter(name) || (backend.ownsWeight() && strings.HasPrefix(name, "flow.w_s_t.")) || (backend.ownsProjection() && projectionParameter(name)) || (backend.ownsSequence() && sequenceParameter(name)) {
			for _, v := range x {
				if v != 0 {
					t.Fatal("resident gradient downloaded", name)
				}
			}
		} else {
			assertSliceClose(t, "resident CPU branch "+name, x, oracle.Gradients[name], 5e-5)
		}
	}
	beforeUploads := d.uploads
	if err := backend.UpdateTrainer(context.Background(), trainer, g, 1e20); err != nil {
		t.Fatal(err)
	}
	expectedUploads := beforeUploads
	if backend.ownsGlobalNorm() {
		expectedUploads++
	}
	if backend.ownsStateCheck() {
		expectedUploads++
	} // reset the post-update diagnostic,not parameter gradients
	if backend.ownsSequence() && expectedUploads != d.uploads {
		t.Fatal("unexpected host parameter-gradient upload")
	}
	if len(d.validatedGradients) != len(trainer.names) {
		t.Fatal("missing optimizer gradient coverage")
	}
	for i, name := range trainer.names {
		assertSliceClose(t, "resident gradient PyTorch "+name, d.validatedGradients[i], oracle.Gradients[name], 5e-5)
	}
	weightDownloads := 0
	if backend.ownsWeight() {
		weightDownloads = 4
	}
	projectionDownloads := 0
	if backend.ownsProjection() {
		projectionDownloads = 12
	}
	sequenceDownloads := 0
	if backend.ownsSequence() {
		sequenceDownloads = 10
	}
	flowBoundary := 4 + 2*len(flow.Time)
	if backend.ownsTime() {
		flowBoundary = 4 + len(flow.Time)
	}
	eosDownloads := 0
	if backend.ownsEOS() {
		eosDownloads = 2
	}
	lossDownloads := 0
	if backend.ownsLoss() {
		lossDownloads = 8
	}
	branchDownloads := 0
	if backend.ownsBranch() {
		branchDownloads = 36
		if backend.ownsStepReduction() {
			branchDownloads = 30
		}
	}
	stepDownloads := 0
	if backend.ownsStepReduction() {
		stepDownloads = 6
	}
	prunedDownloads := 0
	if backend.ownsPruned() {
		prunedDownloads = 32
	}
	if backend.EOSProjectionEnabled() {
		prunedDownloads += 3
	}
	if backend.AudioProjectionEnabled() {
		prunedDownloads += 4
	}
	if backend.VoiceSequenceEnabled() {
		prunedDownloads += 4
	}
	if backend.ConditionerEnabled() {
		prunedDownloads += 2
	}
	if backend.BackboneEnabled() {
		prunedDownloads += 2
	}
	if backend.EOSBackboneEnabled() {
		prunedDownloads++
	}
	if backend.DiagonalEnabled() {
		for _, active := range samples.Mask {
			if active {
				prunedDownloads += 7
			}
		}
	}
	if backend.EndpointEnabled() {
		for _, active := range samples.Mask {
			if active {
				prunedDownloads += 7
			}
		}
	}
	if backend.PrimaryEnabled() {
		for _, active := range samples.Mask {
			if active {
				prunedDownloads += 5
			}
		}
	}
	if backend.DistillEnabled() {
		for _, active := range samples.Mask {
			if active {
				prunedDownloads += 4
			}
		}
	}
	if backend.DistillForwardEnabled() {
		for _, active := range samples.Mask {
			if active {
				prunedDownloads += 2
			}
		}
	}
	if backend.DiagonalForwardEnabled() {
		for _, active := range samples.Mask {
			if active {
				prunedDownloads++
			}
		}
	}
	if backend.weightRootsEnabled() {
		prunedDownloads += 2
	}
	if backend.stepBackboneEnabled() {
		prunedDownloads++
	}
	if backend.stepResultsEnabled() {
		prunedDownloads++
	}
	if backend.distillResultsEnabled() {
		for _, active := range samples.Mask {
			if active {
				prunedDownloads++
			}
		}
	}
	if backend.diagonalAdmissionEnabled() {
		// One scalar guard per valid diagonal row is additional read-only
		// diagnostic transfer, not a downloaded parameter gradient.
		for _, active := range samples.Mask {
			if active {
				prunedDownloads--
			}
		}
	}
	if backend.distillAdmissionEnabled() {
		for _, active := range samples.Mask {
			if active {
				prunedDownloads--
			}
		}
	}
	if backend.eosAdmissionEnabled() {
		prunedDownloads--
	} // additional producer scalar, never a gradient
	if backend.deviceHandoffEnabled() {
		// Two scalar checks per branch replace prior flags; private distill
		// now decodes noise/time explicitly while keeping forward downloads.
		for _, active := range samples.Mask {
			if active {
				prunedDownloads -= 3
			}
		}
	}
	// no forward logit/backward output/logit-seed boundary
	if d.downloads != 2*2+12*flowBoundary+weightDownloads+projectionDownloads+sequenceDownloads+eosDownloads+lossDownloads+branchDownloads+stepDownloads-prunedDownloads {
		t.Fatal("unexpected parameter gradient download", d.downloads)
	}
	for name, want := range oracle.AdamW.ParametersAfterStep {
		assertSliceClose(t, "resident PyTorch update "+name, trainer.params[name], want, 4e-6)
		assertSliceClose(t, "resident PyTorch EMA "+name, trainer.EMA[name], oracle.AdamW.EMAAfterStep[name], 4e-6)
	}
	if backend.active || trainer.StepCount != 1 {
		t.Fatal("completed resident boundary")
	}
	if err := backend.ValidateBoundary(trainer); err != nil {
		t.Fatal(err)
	}
}
func residentGraphRunner(t *testing.T, scope string) (*TrainingRunner, *ResidentTrainingBackend, *hostResidentGraphDriver) {
	t.Helper()
	base := requireRunner(t)
	b, d := installResidentGraph(t, base.trainer, scope)
	id := runIdentityFixture()
	id.Backend = scope
	r, err := NewTrainingRunner(base.trainer, id, base.rows, [6]uint64{1, 2, 3, 4, 5, 6}, base.limits, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	return r, b, d
}
func TestTrainingResidentGraphRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-resident-hybrid-v1", "vulkan-resident-hybrid-v1"} {
		t.Run(scope, func(t *testing.T) {
			a, b, d := residentGraphRunner(t, scope)
			if b.state != nil || d.allocations != 0 {
				t.Fatal("state before restore")
			}
			if _, err := a.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			limits := TrainingBundleLimits{1 << 20, 1 << 16}
			if _, err := a.Checkpoint(dir, "resident", limits); err != nil {
				t.Fatal(err)
			}
			resumed, rb, rd := residentGraphRunner(t, scope)
			if err := resumed.Restore(dir, limits); err != nil {
				t.Fatal(err)
			}
			if rb.state != nil || rd.allocations != 0 {
				t.Fatal("stale resident restore")
			}
			allocations := d.allocations
			for _, r := range []*TrainingRunner{a, resumed} {
				if _, err := r.Step(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			for name, p := range a.trainer.params {
				assertSliceClose(t, "resident graph resume "+name, resumed.trainer.params[name], p, 1e-6)
				assertSliceClose(t, "resident graph resume M "+name, resumed.trainer.Moments[name].M, a.trainer.Moments[name].M, 1e-6)
				assertSliceClose(t, "resident graph resume V "+name, resumed.trainer.Moments[name].V, a.trainer.Moments[name].V, 1e-6)
				assertSliceClose(t, "resident graph resume EMA "+name, resumed.trainer.EMA[name], a.trainer.EMA[name], 1e-6)
			}
			if d.allocations != allocations {
				t.Fatal("resident state reallocated")
			}
			if !bytes.Equal(snapshotRunFixture(t, a.run), snapshotRunFixture(t, resumed.run)) {
				t.Fatal("raw resume drift")
			}
			if err := requireRunner(t).Restore(dir, limits); err == nil {
				t.Fatal("resident restored under SIMD")
			}
			if b.Close() != nil || rb.Close() != nil {
				t.Fatal("close")
			}
		})
	}
}
func TestTrainingResidentGraphFailureBoundaries(t *testing.T) {
	for _, failAt := range []int{1, 2, 7, 14} {
		r, b, d := residentGraphRunner(t, "ptx-resident-hybrid-v1")
		d.failTape = failAt
		before, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
		if _, err := r.Step(context.Background()); err == nil {
			t.Fatal("failed graph accepted", failAt)
		}
		if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !bytes.Equal(raw, snapshotRunFixture(t, r.run)) {
			t.Fatal("failed graph changed host/replay")
		}
		calls := d.tapes
		if _, err := r.Step(context.Background()); err == nil || d.tapes != calls {
			t.Fatal("failed graph retried")
		}
		if _, err := r.Checkpoint(t.TempDir(), "failed", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
			t.Fatal("failed graph checkpoint")
		}
		if b.Close() == nil {
			t.Fatal("failed resident owner freed")
		}
	}
	r, b, _ := residentGraphRunner(t, "ptx-resident-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if b.PrepareTrainingStep(context.Background(), r.trainer) == nil || b.ValidateBoundary(r.trainer) == nil || b.Close() == nil {
		t.Fatal("active boundary guards")
	}
	r.trainer.Flow.TrainingBackend = nil
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("combined owner replaced")
	}
}

func TestTrainingResidentGraphObjectiveWeights(t *testing.T) {
	for _, pEqual := range []float32{0, .25, 1} {
		oracle := loadTrainingStepOracle(t)
		lm, flow, w := trainingStepModelsFromOracle(t, oracle)
		cpuLM, cpuFlow, cpuW := trainingStepModelsFromOracle(t, oracle)
		trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
		if err != nil {
			t.Fatal(err)
		}
		reference, err := NewFullTrainer(cpuLM, cpuFlow, cpuW, DefaultAdamWConfig(), .9)
		if err != nil {
			t.Fatal(err)
		}
		backend, _ := installResidentGraph(t, trainer, "vulkan-resident-hybrid-v1")
		batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
		samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
		config := TrainingStepConfig{PEqual: pEqual, EOSLossWeight: .3}
		if err := backend.PrepareTrainingStep(context.Background(), trainer); err != nil {
			t.Fatal(err)
		}
		got, g, err := PocketTrainingStep(lm, flow, w, batch, samples, config)
		if err != nil {
			t.Fatal(err)
		}
		want, cpuGrad, err := PocketTrainingStep(cpuLM, cpuFlow, cpuW, batch, samples, config)
		if err != nil {
			t.Fatal(err)
		}
		assertClose64(t, "resident weighted loss", got.Loss, want.Loss, 3e-5)
		if err := backend.UpdateTrainer(context.Background(), trainer, g, 1e20); err != nil {
			t.Fatal(err)
		}
		if err := reference.Step(cpuGrad); err != nil {
			t.Fatal(err)
		}
		for name, p := range reference.params {
			assertSliceClose(t, "resident weighted update "+name, trainer.params[name], p, 4e-6)
		}
	}
}
func TestTrainingResidentGraphUpdateAndCheckpointFailures(t *testing.T) {
	for _, at := range []int{1, 2} {
		r, b, d := residentGraphRunner(t, "ptx-resident-hybrid-v1")
		d.failAt = at
		before, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
		if _, err := r.Step(context.Background()); err == nil {
			t.Fatal("failed optimizer update accepted")
		}
		if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !bytes.Equal(raw, snapshotRunFixture(t, r.run)) {
			t.Fatal("failed update changed host or replay")
		}
		if b.Close() == nil {
			t.Fatal("failed update resources released")
		}
		if _, err := r.Checkpoint(t.TempDir(), "invalid", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
			t.Fatal("failed update checkpointed")
		}
	}
	r, b, _ := residentGraphRunner(t, "vulkan-resident-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	r.trainer.params[r.trainer.names[0]][0] += 1
	if _, err := r.Checkpoint(t.TempDir(), "mutated", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
		t.Fatal("mutated boundary checkpointed")
	}
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("mutated host resident update")
	}
	if err := b.ValidateBoundary(r.trainer); err == nil {
		t.Fatal("mutated boundary admitted")
	}
}
func TestTrainingResidentGraphAdmission(t *testing.T) {
	var nilBackend *ResidentTrainingBackend
	if nilBackend.Close() != nil || nilBackend.TrainingOptimizerBackendName() != "unknown" || nilBackend.TrainingFlowBackendName() != "unknown" || nilBackend.TrainingTransformerBackendName() != "unknown" {
		t.Fatal("nil owner")
	}
	base := requireRunner(t)
	b, _ := installResidentGraph(t, base.trainer, "ptx-resident-hybrid-v1")
	if b.PrepareTrainingStep(nil, base.trainer) == nil {
		t.Fatal("nil context")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if b.PrepareTrainingStep(ctx, base.trainer) == nil {
		t.Fatal("cancel")
	}
	b.Scope = "unknown"
	if b.PrepareTrainingStep(context.Background(), base.trainer) == nil {
		t.Fatal("unknown scope")
	}
	b.Scope = "ptx-resident-hybrid-v1"
	driver := b.Driver
	b.Driver = &hostOptimizerDriver{}
	if b.PrepareTrainingStep(context.Background(), base.trainer) == nil {
		t.Fatal("no binding driver")
	}
	b.Driver = driver
	if _, err := b.EvaluateFlow(base.trainer.Flow, nil, nil, nil, 0, nil, nil); err == nil {
		t.Fatal("implicit accumulation")
	}
	if _, err := b.EvaluateTransformer(base.trainer.FlowLM.Transformer, nil, nil, 1); err == nil {
		t.Fatal("implicit accumulation")
	}
	if _, err := b.EvaluateTransformerAccumulating(base.trainer.FlowLM.Transformer, nil, nil, 1, false); err == nil {
		t.Fatal("inactive graph")
	}
	base.trainer.FlowLM.TransformerBackend = nil
	if b.PrepareTrainingStep(context.Background(), base.trainer) == nil {
		t.Fatal("mismatched components")
	}
	base.trainer.FlowLM.TransformerBackend = b
	b.MaxTapeBytes = 0
	if err := b.PrepareTrainingStep(context.Background(), base.trainer); err == nil {
		t.Fatal("zero tape ceiling admitted")
	}
	b.MaxTapeBytes = 1
	if err := b.PrepareTrainingStep(context.Background(), base.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateTransformerAccumulating(base.trainer.FlowLM.Transformer, make([]float32, 4), make([]float32, 4), 1, false); err == nil {
		t.Fatal("tape ceiling")
	}
	if b.Close() == nil {
		t.Fatal("failed tape owner released")
	}
	fresh := &ResidentTrainingBackend{}
	if fresh.ValidateBoundary(base.trainer) != nil {
		t.Fatal("fresh boundary")
	}
}

func TestTrainingResidentGraphInvalidPairedStep(t *testing.T) {
	for _, kind := range []string{"flow-owner", "transformer-owner", "gradient", "ceiling", "mutation", "replacement"} {
		t.Run(kind, func(t *testing.T) {
			r, b, _ := residentGraphRunner(t, "ptx-resident-hybrid-v1")
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			var err error
			switch kind {
			case "flow-owner":
				_, err = b.EvaluateFlowAccumulating(tinyTrainableFlowHead(), nil, nil, nil, 0, nil, nil, false, 0)
			case "transformer-owner":
				_, err = b.EvaluateTransformerAccumulating(tinyFlowLMTraining().Transformer, nil, nil, 1, false)
			case "gradient":
				err = b.UpdateTrainer(context.Background(), r.trainer, TrainingStepGradients{}, 1)
			case "ceiling":
				err = b.UpdateTrainer(context.Background(), r.trainer, TrainingStepGradients{}, 0)
			case "mutation":
				r.trainer.params[r.trainer.names[0]][0] += 1
				err = b.UpdateTrainer(context.Background(), r.trainer, TrainingStepGradients{}, 1)
			case "replacement":
				r.trainer.Flow.TrainingBackend = nil
				_, err = b.EvaluateTransformerAccumulating(r.trainer.FlowLM.Transformer, nil, nil, 1, false)
			}
			if err == nil || b.terminal == nil {
				t.Fatal("invalid pair not terminal")
			}
			if b.Close() == nil {
				t.Fatal("failed pair owner discarded")
			}
		})
	}
}

func TestTrainingResidentWeightRunnerResume(t *testing.T) {
	for _, scope := range []string{"ptx-resident-weight-hybrid-v1", "vulkan-resident-weight-hybrid-v1"} {
		t.Run(scope, func(t *testing.T) {
			a, b, d := residentGraphRunner(t, scope)
			if _, err := a.Step(context.Background()); err != nil {
				t.Fatal(err)
			}
			dir := t.TempDir()
			limits := TrainingBundleLimits{1 << 20, 1 << 16}
			if _, err := a.Checkpoint(dir, "weight", limits); err != nil {
				t.Fatal(err)
			}
			resumed, rb, _ := residentGraphRunner(t, scope)
			if err := resumed.Restore(dir, limits); err != nil {
				t.Fatal(err)
			}
			allocations := d.allocations
			for _, r := range []*TrainingRunner{a, resumed} {
				if _, err := r.Step(context.Background()); err != nil {
					t.Fatal(err)
				}
			}
			for name, p := range a.trainer.params {
				assertSliceClose(t, "resident weighting resume "+name, resumed.trainer.params[name], p, 1e-6)
				assertSliceClose(t, "resident weighting EMA "+name, resumed.trainer.EMA[name], a.trainer.EMA[name], 1e-6)
			}
			if allocations != d.allocations || !bytes.Equal(snapshotRunFixture(t, a.run), snapshotRunFixture(t, resumed.run)) {
				t.Fatal("resident weighting resume allocations/replay")
			}
			legacy, _, _ := residentGraphRunner(t, "ptx-resident-hybrid-v1")
			before := mustFullState(t, legacy.trainer)
			if err := legacy.Restore(dir, limits); err == nil {
				t.Fatal("weight scope restored under legacy resident identity")
			}
			if !reflect.DeepEqual(before, mustFullState(t, legacy.trainer)) {
				t.Fatal("legacy restore mutated")
			}
			if b.Close() != nil || rb.Close() != nil {
				t.Fatal("close")
			}
		})
	}
}
func TestTrainingResidentWeightFailureBoundary(t *testing.T) {
	for _, failure := range []int{1, 15} {
		r, b, d := residentGraphRunner(t, "ptx-resident-weight-hybrid-v1")
		d.failTape = failure
		before, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
		if _, err := r.Step(context.Background()); err == nil {
			t.Fatal("weight tape failure accepted", failure)
		}
		if !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || !bytes.Equal(raw, snapshotRunFixture(t, r.run)) {
			t.Fatal("weighting failure host/replay mutation")
		}
		calls := d.tapes
		if _, err := r.Step(context.Background()); err == nil || calls != d.tapes {
			t.Fatal("weighting retry")
		}
		if b.Close() == nil {
			t.Fatal("weighting retained owner released")
		}
	}
	r, b, _ := residentGraphRunner(t, "vulkan-resident-weight-hybrid-v1")
	r.trainer.Weighting.TrainingBackend = nil
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("weighting removed from named scope")
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err == nil {
		t.Fatal("missing weighting owner")
	}
	legacy, lb, _ := residentGraphRunner(t, "ptx-resident-hybrid-v1")
	legacy.trainer.Weighting.TrainingBackend = lb
	if _, err := legacy.Step(context.Background()); err == nil {
		t.Fatal("weighting injected under legacy identity")
	}
}

func TestTrainingResidentWeightOwnerAdmission(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "vulkan-resident-weight-hybrid-v1")
	if _, err := b.EvaluateWeight(r.trainer.Weighting, nil, nil, 1, false); err == nil {
		t.Fatal("inactive weight backend")
	}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateWeight(tinyLSDWeighting(), nil, nil, 1, false); err == nil || b.terminal == nil {
		t.Fatal("wrong weight model")
	}
	r, b, _ = residentGraphRunner(t, "ptx-resident-weight-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.EvaluateWeight(r.trainer.Weighting, nil, nil, 1, false); err == nil || b.terminal == nil {
		t.Fatal("bad weighting input")
	}
}

// Host mock of specified device F32 global summary merge. Full gradient/update
// assertions use pinned PyTorch or independent CPU gradients, not this mock.
func (d *hostResidentGraphDriver) ValidateNormMerge(ctx context.Context, dst, a, b train.ResidentBuffer, n int, previous int64, mode uint32) error {
	_, err := train.NormMergeShape(n, previous)
	return err
}
func (d *hostResidentGraphDriver) ValidateNormClip(ctx context.Context, dst, a train.ResidentBuffer, elements int64, maxNorm float32) error {
	return train.NormClipShape(elements, maxNorm)
}
func (d *hostResidentGraphDriver) NormMerge(ctx context.Context, dst, aBuffer, bBuffer train.ResidentBuffer, n int, previous int64, mode uint32) error {
	a, b, out := aBuffer.(*hostOptimizerBuffer).values, bBuffer.(*hostOptimizerBuffer).values, dst.(*hostOptimizerBuffer).values
	maximum, sum, invalid, correction := a[0], a[1], a[2], a[3]
	if !finiteF32(a) || invalid != 0 || maximum < 0 || sum < 0 || sum > float32(previous)*(1+2e-6) || (maximum == 0 && sum != 0) {
		invalid = 1
	}
	for i := 0; i < len(b); i += 3 {
		m, s, flag := b[i], b[i+1], b[i+2]
		chunk := min(256, n-(i/3)*256)
		if !finiteF32(b[i:i+3]) || flag != 0 || m < 0 || s < 0 || s > float32(chunk) || (m == 0 && s != 0) || (m > 0 && s < 1) {
			invalid = 1
		}
		if mode == 0 {
			maximum = max(maximum, m)
		} else if m > maximum {
			invalid = 1
		}
	}
	total, comp := sum, correction
	if mode == 0 {
		total, comp = 0, 0
	} else if invalid == 0 && maximum > 0 {
		for i := 0; i < len(b); i += 3 {
			r := b[i] / maximum
			term := (r*r)*b[i+1] - comp
			next := total + term
			comp = (next - total) - term
			total = next
		}
	}
	out[0], out[1], out[2], out[3] = maximum, total, invalid, comp
	return ctx.Err()
}
func (d *hostResidentGraphDriver) NormClip(ctx context.Context, dst, a train.ResidentBuffer, elements int64, maxNorm float32) error {
	x, out := a.(*hostOptimizerBuffer).values, dst.(*hostOptimizerBuffer).values
	scale := float32(1)
	if x[2] == 0 && x[0] > 0 {
		root := float32(math.Sqrt(float64(x[1])))
		ratio := maxNorm / x[0]
		if ratio < root {
			scale = ratio / root
		}
	}
	if x[2] != 0 {
		scale = 0
	}
	copy(out, x)
	out[3] = scale
	return ctx.Err()
}
