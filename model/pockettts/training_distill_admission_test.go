//go:build linux

package pockettts

import (
	"context"
	"encoding/json"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"strings"
	"testing"
)

func distillAdmissionFixture(t *testing.T) DistillForwardTape {
	t.Helper()
	m := tinyTrainableFlowHead()
	in := primaryFixture()
	p, err := m.BuildDistillForwardTape(DistillForwardInputs{Condition: in.Condition, Target: in.Target, Noise: in.Noise, S: .3, T: .8}, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func TestTrainingDistillAdmissionTape(t *testing.T) {
	source := distillAdmissionFixture(t)
	before, _ := json.Marshal(source)
	a, err := appendDistillAdmission(source, .1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	b, err := appendDistillAdmission(source, .1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Tape.Tensors) != len(source.Tape.Tensors)+2 || len(a.Tape.Commands) != len(source.Tape.Commands)+1 || len(a.Tape.Outputs) != len(source.Tape.Outputs)+1 {
		t.Fatal("guard counts")
	}
	c := a.Tape.Commands[len(a.Tape.Commands)-1]
	if c.Op != "lsd-loss" || c.Mode != 2 || c.Scale != .1 || c.Inputs[0] != source.Tape.Outputs[5] || c.Inputs[1] != source.Tape.Outputs[4] {
		t.Fatal("prediction/desired/logvar edges")
	}
	values := interpretFlowTape(t, a.Tape)
	legacy := interpretFlowTape(t, source.Tape)
	for i := 0; i < 6; i++ {
		assertSliceClose(t, "public distillation unchanged", values[i], legacy[i], 2e-6)
	}
	selected := append(append([][]float32(nil), values[:6]...), values[len(values)-1])
	if validateDistillAdmissionFlag(selected, DistillForwardInputs{}) != nil {
		t.Fatal("flag")
	}
	a.Tape.Commands[len(a.Tape.Commands)-1].Inputs[0] = -1
	a.Tape.Tensors[len(a.Tape.Tensors)-2].Values[0] = 0
	a.Tape.Outputs[0] = -1
	after, _ := json.Marshal(source)
	if string(before) != string(after) || b.Tape.Commands[len(b.Tape.Commands)-1].Inputs[0] != source.Tape.Outputs[5] {
		t.Fatal("source/result ownership")
	}
	for _, lv := range []float32{-81, 81, float32(math.NaN())} {
		if _, err := appendDistillAdmission(source, lv, 64<<20); err == nil {
			t.Fatal("invalid logvar")
		}
	}
	if _, err := appendDistillAdmission(source, .1, tapeLogicalBytes(source.Tape)); err == nil {
		t.Fatal("expanded budget")
	}
	bad := source
	bad.Tape.Tensors = append(append([]train.TensorSpec(nil), source.Tape.Tensors...), train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if _, err := appendDistillAdmission(bad, 0, 64<<20); err == nil {
		t.Fatal("dead source")
	}
	minimal := DistillForwardTape{Tape: train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{1}}, {Shape: []int{1}, Values: []float32{0}}}, Outputs: []int{1}}, ResidentBindings: map[string]train.ResidentTapeBinding{"p": {Parameter: 0, Gradient: 1}}}
	if _, err := appendDistillAdmission(minimal, 0, 1024); err == nil || !strings.Contains(err.Error(), "source outputs") {
		t.Fatal("outputs", err)
	}
	minimal.Tape.Outputs = []int{1, 1, 1, 1, 1, 1}
	if _, err := appendDistillAdmission(minimal, 0, 16); err == nil || !strings.Contains(err.Error(), "loss tape byte ceiling") {
		t.Fatal("standalone budget", err)
	}
}
func TestTrainingDistillAdmissionFlag(t *testing.T) {
	for _, kind := range []string{"count", "shape", "flag", "nan", "alias", "borrowed"} {
		values := [][]float32{{1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {0}}
		in := DistillForwardInputs{}
		switch kind {
		case "count":
			values = values[:6]
		case "shape":
			values[6] = nil
		case "flag":
			values[6][0] = 1
		case "nan":
			values[6][0] = float32(math.NaN())
		case "alias":
			values[6] = values[0][:1]
			values[0][0] = 0
		case "borrowed":
			in.Target = values[6]
		}
		if validateDistillAdmissionFlag(values, in) == nil {
			t.Fatal("bad flag", kind)
		}
	}
}
func TestTrainingResidentDistillAdmissionDirectGuard(t *testing.T) {
	for _, scope := range []string{"ptx-resident-distilladmission-hybrid-v1", "vulkan-resident-distilladmission-hybrid-v1"} {
		r, b, d := residentGraphRunner(t, scope)
		m := r.trainer.Flow
		in := DistillForwardInputs{Condition: make([]float32, m.Condition.In), Target: make([]float32, m.Input.In), Noise: make([]float32, m.Input.In), S: .3, T: .8}
		if _, err := b.evaluateDistillAdmitted(m, in, .1); err == nil {
			t.Fatal("inactive")
		}
		if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
			t.Fatal(err)
		}
		before := d.accumulations
		got, err := b.evaluateDistillAdmitted(m, in, .1)
		if err != nil {
			t.Fatal(err)
		}
		public, err := b.EvaluateDistillForward(m, in)
		if err != nil {
			t.Fatal(err)
		}
		for _, pair := range [][2][]float32{{got.PrimaryInput, public.PrimaryInput}, {got.Velocity, public.Velocity}, {got.Tangent, public.Tangent}, {got.EndpointInput, public.EndpointInput}, {got.Desired, public.Desired}, {got.Prediction, public.Prediction}} {
			assertSliceClose(t, "distill admitted field", pair[0], pair[1], 2e-6)
		}
		if d.accumulations != before {
			t.Fatal("forward accumulated")
		}
		if b.Close() == nil {
			t.Fatal("active close")
		}
		if b.state.Close() != nil {
			t.Fatal("mock cleanup")
		}
		b.active = false
		if b.Close() != nil {
			t.Fatal("close")
		}
	}
}
func TestTrainingResidentDistillAdmissionRejectOrdering(t *testing.T) {
	for _, kind := range []string{"flag", "shape", "nan", "alias", "hostoverflow", "vectorshape"} {
		r, b, d := residentGraphRunner(t, "ptx-resident-distilladmission-hybrid-v1")
		seen := false
		before := 0
		d.afterDownload = func(values [][]float32) {
			if len(values) != 7 || len(values[6]) != 1 {
				return
			}
			seen = true
			before = d.accumulations
			switch kind {
			case "flag":
				values[6][0] = 1
			case "shape":
				values[6] = nil
			case "nan":
				values[6][0] = float32(math.NaN())
			case "alias":
				values[6] = values[0][:1]
				values[0][0] = 0
			case "hostoverflow":
				values[5][0] = math.MaxFloat32
			case "vectorshape":
				values[2] = nil
			}
		}
		state, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
		if _, err := r.Step(context.Background()); err == nil {
			t.Fatal("bad distill guard", kind)
		}
		if !seen || d.accumulations != before || !reflect.DeepEqual(state, mustFullState(t, r.trainer)) || string(raw) != string(snapshotRunFixture(t, r.run)) {
			t.Fatal("distill backward/publication after guard", kind)
		}
		tapes := d.tapes
		if _, err := r.Step(context.Background()); err == nil || d.tapes != tapes || b.Close() == nil {
			t.Fatal("retry/owner")
		}
	}
}
func TestTrainingResidentDistillAdmissionOwnerAndBudget(t *testing.T) {
	for _, kind := range []string{"owner", "logvar", "budget", "forward"} {
		r, b, d := residentGraphRunner(t, "ptx-resident-distilladmission-hybrid-v1")
		if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
			t.Fatal(err)
		}
		m := r.trainer.Flow
		in := DistillForwardInputs{Condition: make([]float32, m.Condition.In), Target: make([]float32, m.Input.In), Noise: make([]float32, m.Input.In), S: .3, T: .8}
		lv := float32(.1)
		switch kind {
		case "owner":
			m = &FlowHeadCPU{}
		case "logvar":
			lv = 81
		case "budget":
			b.MaxTapeBytes = 1
		case "forward":
			in.S = -1
		}
		tapes := d.tapes
		if _, err := b.evaluateDistillAdmitted(m, in, lv); err == nil || b.terminal == nil || d.tapes != tapes {
			t.Fatal("guard admission", kind)
		}
	}
	r, b, _ := residentGraphRunner(t, "ptx-resident-diagadmission-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.evaluateDistillAdmitted(r.trainer.Flow, DistillForwardInputs{}, 0); err == nil || !strings.Contains(err.Error(), "scope") {
		t.Fatal("oldscope injected", err)
	}
}
func TestTrainingResidentDistillAdmissionBorrowedFlag(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-distilladmission-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	m := r.trainer.Flow
	in := DistillForwardInputs{Condition: make([]float32, m.Condition.In), Target: make([]float32, m.Input.In), Noise: make([]float32, m.Input.In), S: .3, T: .8}
	d.afterDownload = func(values [][]float32) { values[6] = in.Target[:1] }
	if _, err := b.evaluateDistillAdmitted(m, in, 0); err == nil || !strings.Contains(err.Error(), "borrowed flag") || b.terminal == nil {
		t.Fatal("borrowed scalar", err)
	}
}
func TestTrainingResidentDistillAdmissionAccounting(t *testing.T) {
	old, ob, a := residentGraphRunner(t, "ptx-resident-diagadmission-hybrid-v1")
	next, nb, b := residentGraphRunner(t, "ptx-resident-distilladmission-hybrid-v1")
	if _, err := old.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if _, err := next.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if b.tapes != a.tapes || b.graphCommands != a.graphCommands+2 || b.downloads != a.downloads+2 || b.downloadedElements != a.downloadedElements+2 || b.accumulations != a.accumulations {
		t.Fatal("guard accounting")
	}
	t.Logf("tiny tapes %d/%d commands %d/%d downloads %d/%d elements %d/%d logical %d/%d plannedstorage %d/%d", a.tapes, b.tapes, a.graphCommands, b.graphCommands, a.downloads, b.downloads, a.downloadedElements, b.downloadedElements, a.graphBytes, b.graphBytes, a.storageBytes, b.storageBytes)
	if ob.Close() != nil || nb.Close() != nil {
		t.Fatal("close")
	}
}
func TestTrainingResidentDistillAdmissionCrossDiagonalRestore(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-distilladmission-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	limits := TrainingBundleLimits{1 << 20, 1 << 16}
	if _, err := r.Checkpoint(dir, "guard", limits); err != nil {
		t.Fatal(err)
	}
	old, ob, _ := residentGraphRunner(t, "ptx-resident-diagadmission-hybrid-v1")
	before := mustFullState(t, old.trainer)
	if old.Restore(dir, limits) == nil || !reflect.DeepEqual(before, mustFullState(t, old.trainer)) {
		t.Fatal("crossscope restore")
	}
	if b.Close() != nil || ob.Close() != nil {
		t.Fatal("close")
	}
}

func TestTrainingResidentDistillAdmissionPublicPolicy(t *testing.T) {
	for _, private := range []bool{false, true} {
		r, b, d := residentGraphRunner(t, "ptx-resident-distilladmission-hybrid-v1")
		if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
			t.Fatal(err)
		}
		m := r.trainer.Flow
		condition := make([]float32, m.Condition.In)
		target, noise := make([]float32, m.Input.In), make([]float32, m.Input.In)
		guards := 0
		d.afterDownload = func(values [][]float32) {
			if len(values) == 7 && len(values[6]) == 1 {
				guards++
			}
		}
		_, _, err := m.lsdDistillTrainingBackendResults(b, condition, .3, .8, noise, target, .1, 1, private)
		if err != nil {
			t.Fatal(err)
		}
		if private && guards != 1 || !private && guards != 0 {
			t.Fatal("public/private guard policy", private, guards)
		}
		if b.Close() == nil {
			t.Fatal("active close")
		}
		if b.state.Close() != nil {
			t.Fatal("mock cleanup")
		}
		b.active = false
		if b.Close() != nil {
			t.Fatal("close")
		}
	}
}

func TestTrainingResidentDistillAdmissionLateRowFailure(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-distilladmission-hybrid-v1")
	guards, atFailure := 0, 0
	d.afterDownload = func(values [][]float32) {
		if len(values) == 7 && len(values[6]) == 1 {
			guards++
			if guards == 2 {
				atFailure = d.accumulations
				values[6][0] = 1
			}
		}
	}
	state, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
	metrics, err := r.Step(context.Background())
	if err == nil || guards != 2 || atFailure == 0 || d.accumulations != atFailure {
		t.Fatal("late row failure", err, guards, atFailure)
	}
	if metrics != (TrainingStepMetrics{}) || d.updates != 0 || len(d.validatedGradients) != 0 || !reflect.DeepEqual(state, mustFullState(t, r.trainer)) || string(raw) != string(snapshotRunFixture(t, r.run)) {
		t.Fatal("late failure published update/metrics/replay")
	}
	if r.poison == nil || b.terminal == nil || b.state == nil {
		t.Fatal("failed owner not terminal")
	}
	// Earlier row gradients may remain device-resident; they cannot be converted
	// into a completed boundary, exported or reused after the row guard fails.
	if _, err := r.Checkpoint(t.TempDir(), "failed", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
		t.Fatal("checkpoint after failed row")
	}
	if _, err := b.state.Snapshot(context.Background()); err == nil {
		t.Fatal("dirty/failed state snapshot")
	}
	tapes := d.tapes
	if _, err := r.Step(context.Background()); err == nil || d.tapes != tapes || b.Close() == nil {
		t.Fatal("retry/release")
	}
}
