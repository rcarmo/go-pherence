package pockettts

import (
	"encoding/json"
	"errors"
	"os"
	"testing"
)

type hostFlowTapeExecutor struct{ t *testing.T }

func (e hostFlowTapeExecutor) ExecuteFlowTape(tape FlowDualTrainingTape) (FlowDualTrainingResult, error) {
	return tape.Decode(interpretFlowTape(e.t, tape.Tape))
}
func TestTrainingFlowBackendLSDPyTorchParity(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowHeadBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit {
		t.Fatal("oracle provenance")
	}
	model := flowHeadFromOracle(t, oracle.Parameters)
	want := oracle.LSDMinimal
	backend := &FlowTapeTrainingBackend{Executor: hostFlowTapeExecutor{t}, MaxBytes: 64 << 20}
	result, g, err := model.LSDDistillTrainingBackend(backend, want.Condition, want.S, want.T, want.Noise, want.Target, want.LogVariance)
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "backend LSD loss", result.Loss, float64(want.Loss), 2e-6)
	assertClose64(t, "backend LSD raw square", result.RawSquare, float64(want.RawSquare), 2e-6)
	assertSliceClose(t, "backend LSD velocity", result.Velocity, want.Velocity, 5e-6)
	assertSliceClose(t, "backend LSD JVP", result.TimeDerivative, want.TimeDerivative, 5e-6)
	assertSliceClose(t, "backend LSD endpoint", result.Endpoint, want.Endpoint, 5e-6)
	assertSliceClose(t, "backend LSD residual", result.Residual, want.Residual, 5e-6)
	assertSliceClose(t, "backend LSD dCondition", g.Condition, want.DCondition, 2e-5)
	assertSliceClose(t, "backend LSD dNoise", g.Noise, want.DNoise, 2e-5)
	assertSliceClose(t, "backend LSD dTarget", g.Target, want.DTarget, 2e-5)
	assertClose64(t, "backend LSD ds", float64(g.DS), float64(want.DS), 2e-5)
	assertClose64(t, "backend LSD dt", float64(g.DT), float64(want.DT), 2e-5)
	assertClose64(t, "backend LSD dlogvar", float64(g.DLogVariance), float64(want.DLogVariance), 2e-5)
	for name, values := range flowHeadGradientMap(g.Flow) {
		assertSliceClose(t, "backend LSD PyTorch "+name, values, want.Gradients[name], 3e-5)
	}
}

type failingFlowBackend struct{ err error }

func (b failingFlowBackend) EvaluateFlow(*FlowHeadCPU, []float32, []float32, []float32, int, []float32, []float32) (FlowDualTrainingResult, error) {
	return FlowDualTrainingResult{}, b.err
}
func TestTrainingFlowTapeAdmissionErrors(t *testing.T) {
	model := tinyTrainableFlowHead()
	condition, times, x, gv, gt := []float32{.2, -.3, .5}, []float32{.25, .8}, []float32{.3, -.2}, []float32{.1, .2}, []float32{.3, -.4}
	if _, err := model.BuildFlowDualTrainingTape(condition, times, x, 1, gv, gt, 1); err == nil {
		t.Fatal("budget admitted")
	}
	if _, err := model.BuildFlowDualTrainingTape(condition, times, x, 2, gv, gt, 1<<20); err == nil {
		t.Fatal("time index admitted")
	}
	sentinel := errors.New("executor failed")
	if _, _, err := model.LSDDistillTrainingBackend(failingFlowBackend{sentinel}, condition, .25, .8, x, gv, .1); !errors.Is(err, sentinel) {
		t.Fatal(err)
	}
	if _, _, err := model.LSDDistillTrainingBackend(nil, condition, .25, .8, x, gv, .1); err == nil {
		t.Fatal("nil backend")
	}
	for _, mutate := range []func(*FlowHeadCPU){func(m *FlowHeadCPU) { m.Time = m.Time[:1] }, func(m *FlowHeadCPU) { m.Final.Linear.Out = 1 }, func(m *FlowHeadCPU) { m.Condition.In = 4 }} {
		bad := *model
		mutate(&bad)
		if _, _, err := bad.LSDDistillTrainingBackend(failingFlowBackend{sentinel}, condition, .25, .8, x, gv, .1); err == nil || errors.Is(err, sentinel) {
			t.Fatal("invalid topology reached backend", err)
		}
	}
}
