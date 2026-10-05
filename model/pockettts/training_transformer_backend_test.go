package pockettts

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"testing"
)

type hostTransformerTapeExecutor struct {
	t      *testing.T
	calls  int
	failAt int
	err    error
}

func (e *hostTransformerTapeExecutor) ExecuteTransformerTape(tape TransformerTrainingTape) (TransformerTrainingResult, error) {
	e.calls++
	if e.failAt == e.calls {
		return TransformerTrainingResult{}, e.err
	}
	return tape.Decode(interpretFlowTape(e.t, tape.Tape))
}
func (e *hostTransformerTapeExecutor) Close() error { return e.err }
func TestTrainingTransformerBackendFlowLMPyTorch(t *testing.T) {
	data, err := os.ReadFile("testdata/flowlm_training_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowLMTrainingOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit {
		t.Fatal("oracle provenance")
	}
	m := flowLMTrainingFromOracle(t, oracle)
	executor := &hostTransformerTapeExecutor{t: t}
	m.TransformerBackend = &TransformerTapeTrainingBackend{Executor: executor, MaxBytes: 64 << 20}
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	out, g, input, err := m.ForwardBackward(batch, oracle.DZ, oracle.DEOS)
	if err != nil {
		t.Fatal(err)
	}
	if executor.calls != 2 {
		t.Fatal("transformer evaluation count", executor.calls)
	}
	assertSliceClose(t, "transformer backend z", out.Z, oracle.Z, 5e-6)
	assertSliceClose(t, "transformer backend eos", out.EOS, oracle.EOS, 5e-6)
	assertSliceClose(t, "transformer backend audio", input.NormalizedLatents, oracle.DNormalizedLatents, 1.5e-5)
	assertSliceClose(t, "transformer backend voice", input.VoiceLatents, oracle.DVoiceLatents, 1.5e-5)
	for name, values := range flowLMTrainingGradientMap(g) {
		assertSliceClose(t, "transformer backend FlowLM PyTorch "+name, values, oracle.Gradients[name], 2e-5)
	}
}
func TestTrainingTransformerBackendFullStepPyTorch(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	lm.TransformerBackend = &TransformerTapeTrainingBackend{Executor: &hostTransformerTapeExecutor{t: t}, MaxBytes: 64 << 20}
	flow.TrainingBackend = &FlowTapeTrainingBackend{Executor: hostFlowTapeExecutor{t}, MaxBytes: 64 << 20}
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime, DistillS: oracle.DistillS, DistillT: oracle.DistillT}
	metrics, g, err := PocketTrainingStep(lm, flow, w, batch, samples, DefaultTrainingStepConfig())
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "transformer+flow tape full loss", metrics.Loss, oracle.Metrics["loss"], 3e-5)
	for name, values := range trainingStepGradientMap(g) {
		assertSliceClose(t, "transformer+flow tape full PyTorch "+name, values, oracle.Gradients[name], 5e-5)
	}
	assertSliceClose(t, "transformer+flow target", g.Inputs.NormalizedLatents, oracle.DNormalizedLatents, 5e-5)
	assertSliceClose(t, "transformer+flow voice", g.Inputs.VoiceLatents, oracle.DVoiceLatents, 5e-5)
}
func TestTrainingTransformerBackendAdmission(t *testing.T) {
	m := tinyFlowLMTraining()
	x, seed := make([]float32, 3*m.Hidden), make([]float32, 3*m.Hidden)
	executor := &hostTransformerTapeExecutor{t: t}
	backend := &TransformerTapeTrainingBackend{Executor: executor, MaxBytes: 64 << 20}
	if _, _, _, err := m.Transformer.ForwardBackwardTrainingBackend(x, seed, 3, backend); err != nil {
		t.Fatal(err)
	}
	for _, b := range []*TransformerTapeTrainingBackend{nil, {}} {
		if _, err := b.EvaluateTransformer(m.Transformer, x, seed, 3); err == nil {
			t.Fatal("nil executor")
		}
		if b.Close() != nil {
			t.Fatal("nil close")
		}
	}
	if _, _, _, err := m.Transformer.ForwardBackwardTrainingBackend(x, seed, 3, nil); err == nil {
		t.Fatal("nil backend")
	}
	if _, _, _, err := m.Transformer.ForwardBackwardTrainingBackend(nil, seed, 3, backend); err == nil {
		t.Fatal("short sequence")
	}
	short := &TransformerTapeTrainingBackend{Executor: executor, MaxBytes: 1}
	if _, err := short.EvaluateTransformer(m.Transformer, x, seed, 3); err == nil {
		t.Fatal("over budget")
	}
	sentinel := errors.New("executor failed")
	executor.err = sentinel
	if !errors.Is(backend.Close(), sentinel) {
		t.Fatal("close error")
	}
	for _, failAt := range []int{1, 2} {
		executor = &hostTransformerTapeExecutor{t: t, failAt: failAt, err: sentinel}
		m.TransformerBackend = &TransformerTapeTrainingBackend{Executor: executor, MaxBytes: 64 << 20}
		if _, _, _, err := m.ForwardBackward(tinyFlowLMBatch(), make([]float32, 3*m.Hidden), make([]float32, 3)); !errors.Is(err, sentinel) {
			t.Fatal("backend failure lost", err)
		}
		if executor.calls != failAt {
			t.Fatal("failure fallback")
		}
	}
	m.TransformerBackend = backend
	m.AttentionBackwardBackend = attentionBackendFunc(referenceAttentionBackend)
	if _, _, _, err := m.ForwardBackward(tinyFlowLMBatch(), make([]float32, 3*m.Hidden), make([]float32, 3)); err == nil {
		t.Fatal("combined backend")
	}
	r := requireRunner(t)
	r.trainer.FlowLM.TransformerBackend = backend
	if _, err := r.Step(context.Background()); err == nil {
		t.Fatal("transformer saved under SIMD scope")
	}
	if _, err := NewTrainingRunner(r.trainer, runIdentityFixture(), r.rows, [6]uint64{1, 2, 3, 4, 5, 6}, r.limits, 1, true); err == nil {
		t.Fatal("unbound runner backend")
	}
}
func TestTrainingTransformerTapeRejectedInputs(t *testing.T) {
	m := tinyFlowLMTraining().Transformer
	x, seed := make([]float32, 3*m.Width), make([]float32, 3*m.Width)
	if _, err := m.BuildTransformerTrainingTape(x, seed, 3, 0); err == nil {
		t.Fatal("budget")
	}
	if _, err := m.BuildTransformerTrainingTape(x, seed, 3, 1); err == nil {
		t.Fatal("peak admission")
	}
	if _, err := m.BuildTransformerTrainingTape(nil, seed, 3, 64<<20); err == nil {
		t.Fatal("input length")
	}
	layers := m.Layers
	m.Layers = make([]TransformerLayerCPU, 17)
	for i := range m.Layers {
		m.Layers[i] = layers[0]
	}
	if _, err := m.BuildTransformerTrainingTape(x, seed, 3, 64<<20); err == nil {
		t.Fatal("layer ceiling")
	}
	m.Layers = layers
	old := m.Layers[0].Norm1Weight[0]
	m.Layers[0].Norm1Weight[0] = float32(math.NaN())
	if _, err := m.BuildTransformerTrainingTape(x, seed, 3, 64<<20); err == nil {
		t.Fatal("nonfinite parameter")
	}
	m.Layers[0].Norm1Weight[0] = old
	tape, err := m.BuildTransformerTrainingTape(x, seed, 3, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	values := interpretFlowTape(t, tape.Tape)
	if _, err := tape.Decode(nil); err == nil {
		t.Fatal("output count")
	}
	values[0][0] = float32(math.NaN())
	if _, err := tape.Decode(values); err == nil {
		t.Fatal("nonfinite output")
	}
	var zero TransformerTrainingTape
	if _, err := zero.Decode(nil); err == nil {
		t.Fatal("empty tape")
	}
}

func TestTrainingTransformerBackendResultRejection(t *testing.T) {
	m := tinyFlowLMTraining().Transformer
	x, seed := make([]float32, 3*m.Width), make([]float32, 3*m.Width)
	backend := &TransformerTapeTrainingBackend{Executor: &hostTransformerTapeExecutor{t: t}, MaxBytes: 64 << 20}
	fresh := func() TransformerTrainingResult {
		r, err := backend.EvaluateTransformer(m, x, seed, 3)
		if err != nil {
			t.Fatal(err)
		}
		return r
	}
	for _, mutate := range []func(*TransformerTrainingResult){func(r *TransformerTrainingResult) { r.Output = nil }, func(r *TransformerTrainingResult) { r.Output[0] = float32(math.NaN()) }, func(r *TransformerTrainingResult) { r.DSequence = r.Output }, func(r *TransformerTrainingResult) { r.Output = x }, func(r *TransformerTrainingResult) { r.Parameters = nil }, func(r *TransformerTrainingResult) {
		for key := range r.Parameters {
			r.Parameters[key] = nil
			break
		}
	}, func(r *TransformerTrainingResult) {
		for key := range r.Parameters {
			r.Parameters[key][0] = float32(math.Inf(1))
			break
		}
	}} {
		r := fresh()
		mutate(&r)
		if validateTransformerBackendResult(m, r, x, seed, 3) == nil {
			t.Fatal("invalid result admitted")
		}
	}
	for _, named := range []map[string][]float32{nil, {"bad": nil}} {
		if _, err := transformerGradientsFromNamed(m, named); err == nil {
			t.Fatal("invalid named gradients")
		}
	}
	r := fresh()
	for key := range r.Parameters {
		r.Parameters[key] = nil
		break
	}
	if _, err := transformerGradientsFromNamed(m, r.Parameters); err == nil {
		t.Fatal("invalid gradient shape")
	}
	m.FinalWeight, m.FinalBias = nil, nil
	if _, _, _, err := m.ForwardBackwardTrainingBackend(x, seed, 3, backend); err != nil {
		t.Fatal("optional final norm", err)
	}
	for _, executor := range []TransformerTapeExecutor{(*PTXFlowTapeExecutor)(nil), (*VulkanFlowTapeExecutor)(nil)} {
		if _, err := executor.ExecuteTransformerTape(TransformerTrainingTape{}); err == nil {
			t.Fatal("nil native owner")
		}
	}
}

func TestTrainingTransformerBackendCapturedForward(t *testing.T) {
	m := tinyFlowLMTraining()
	e := &hostTransformerTapeExecutor{t: t}
	backend := &TransformerTapeTrainingBackend{Executor: e, MaxBytes: 64 << 20}
	m.TransformerBackend = backend
	batch := tinyFlowLMBatch()
	dz, deos := make([]float32, 3*m.Hidden), make([]float32, 3)
	tape, err := m.forwardTrainingTape(batch, dz, deos)
	if err != nil {
		t.Fatal(err)
	}
	m.TransformerBackend = nil
	if _, _, err := m.backwardTrainingTape(tape, batch, dz, deos); err == nil {
		t.Fatal("removed backend accepted")
	}
	m.TransformerBackend = backend
	m.AttentionBackwardBackend = attentionBackendFunc(referenceAttentionBackend)
	if _, _, err := m.backwardTrainingTape(tape, batch, dz, deos); err == nil {
		t.Fatal("combined backend accepted")
	}
	m.AttentionBackwardBackend = nil
	m.TransformerBackend = nil
	tape, err = m.forwardTrainingTape(batch, dz, deos)
	if err != nil {
		t.Fatal(err)
	}
	m.TransformerBackend = backend
	if _, _, err := m.backwardTrainingTape(tape, batch, dz, deos); err == nil {
		t.Fatal("backend injected after CPU forward")
	}
	var nilBackend *TransformerTapeTrainingBackend
	if nilBackend.TrainingTransformerBackendName() != "unknown" {
		t.Fatal("typed nil identity")
	}
	if (&TransformerTapeTrainingBackend{}).TrainingTransformerBackendName() != "unknown" {
		t.Fatal("unnamed identity")
	}
	var ptx *PTXFlowTapeExecutor
	var vk *VulkanFlowTapeExecutor
	if ptx.TrainingTransformerBackendName() != "unknown" || vk.TrainingTransformerBackendName() != "unknown" {
		t.Fatal("nil native identity")
	}
	if (&TransformerTapeTrainingBackend{Executor: &PTXFlowTapeExecutor{}}).TrainingTransformerBackendName() != "ptx-transformer-hybrid-v1" || (&TransformerTapeTrainingBackend{Executor: &VulkanFlowTapeExecutor{}}).TrainingTransformerBackendName() != "vulkan-transformer-hybrid-v1" {
		t.Fatal("native scope identity")
	}
}
