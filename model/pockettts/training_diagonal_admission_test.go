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

func diagonalAdmissionFixture(t *testing.T) DiagonalForwardTape {
	t.Helper()
	m := tinyTrainableFlowHead()
	in := primaryFixture()
	p, err := m.BuildDiagonalForwardTape(DiagonalForwardInputs{Condition: in.Condition, Target: in.Target, Noise: in.Noise, Time: .3}, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	return p
}
func TestTrainingDiagonalAdmissionTape(t *testing.T) {
	p := diagonalAdmissionFixture(t)
	before, _ := json.Marshal(p)
	a, err := appendDiagonalAdmission(p, .1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	b, err := appendDiagonalAdmission(p, .1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Tape.Commands) != len(p.Tape.Commands)+1 || len(a.Tape.Tensors) != len(p.Tape.Tensors)+2 || len(a.Tape.Outputs) != len(p.Tape.Outputs)+1 {
		t.Fatal("guard accounting")
	}
	command := a.Tape.Commands[len(a.Tape.Commands)-1]
	if command.Op != "lsd-loss" || command.Mode != 2 || command.Scale != .1 || command.Inputs[0] != p.Tape.Outputs[2] || command.Inputs[1] != p.Tape.Outputs[1] {
		t.Fatal("guard source IDs")
	}
	values := interpretFlowTape(t, a.Tape)
	legacy := interpretFlowTape(t, p.Tape)
	for i := 0; i < 3; i++ {
		assertSliceClose(t, "public diagonal unchanged", values[i], legacy[i], 2e-6)
	}
	selected := [][]float32{values[0], values[1], values[2], values[len(values)-1]}
	if err := validateDiagonalAdmissionFlag(selected); err != nil {
		t.Fatal(err)
	}
	a.Tape.Commands[len(a.Tape.Commands)-1].Inputs[0] = -1
	a.Tape.Tensors[len(a.Tape.Tensors)-2].Values[0] = 0
	a.Tape.Outputs[0] = -1
	after, _ := json.Marshal(p)
	if string(before) != string(after) || b.Tape.Commands[len(b.Tape.Commands)-1].Inputs[0] != p.Tape.Outputs[2] {
		t.Fatal("source/result ownership")
	}
	for _, lv := range []float32{-81, 81, float32(math.NaN())} {
		if _, err := appendDiagonalAdmission(p, lv, 64<<20); err == nil {
			t.Fatal("invalid logvar")
		}
	}
	if _, err := appendDiagonalAdmission(p, .1, tapeLogicalBytes(p.Tape)); err == nil {
		t.Fatal("expanded budget")
	}
	bad := p
	bad.Tape.Tensors = append(append([]train.TensorSpec(nil), p.Tape.Tensors...), train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	if _, err := appendDiagonalAdmission(bad, .1, 64<<20); err == nil {
		t.Fatal("deadsource")
	}
}
func TestTrainingDiagonalAdmissionFlag(t *testing.T) {
	makeValues := func() [][]float32 { return [][]float32{{1, 2}, {3, 4}, {5, 6}, {0}} }
	for _, kind := range []string{"count", "shape", "flag", "nan", "alias"} {
		v := makeValues()
		switch kind {
		case "count":
			v = v[:3]
		case "shape":
			v[3] = nil
		case "flag":
			v[3][0] = 1
		case "nan":
			v[3][0] = float32(math.NaN())
		case "alias":
			v[3] = v[0][:1]
			v[0][0] = 0
		}
		if validateDiagonalAdmissionFlag(v) == nil {
			t.Fatal("bad flag", kind)
		}
	}
}
func TestTrainingResidentDiagonalAdmissionDirectGuard(t *testing.T) {
	for _, scope := range []string{"ptx-resident-diagadmission-hybrid-v1", "vulkan-resident-diagadmission-hybrid-v1"} {
		r, b, d := residentGraphRunner(t, scope)
		if _, err := b.evaluateDiagonalAdmitted(r.trainer.Flow, DiagonalForwardInputs{}, 0); err == nil {
			t.Fatal("inactive")
		}
		if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
			t.Fatal(err)
		}
		input := DiagonalForwardInputs{Condition: make([]float32, r.trainer.Flow.Condition.In), Target: make([]float32, r.trainer.Flow.Input.In), Noise: make([]float32, r.trainer.Flow.Input.In), Time: .3}
		before := d.accumulations
		got, err := b.evaluateDiagonalAdmitted(r.trainer.Flow, input, .1)
		if err != nil {
			t.Fatal(err)
		}
		if d.accumulations != before {
			t.Fatal("guard accumulated")
		}
		public, err := b.EvaluateDiagonalForward(r.trainer.Flow, input)
		if err != nil {
			t.Fatal(err)
		}
		assertSliceClose(t, "admitted forward", got.Prediction, public.Prediction, 2e-6)
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
func TestTrainingResidentDiagonalAdmissionRejectOrdering(t *testing.T) {
	for _, kind := range []string{"invalidflag", "flagshape", "flagnan", "flagalias", "hostoverflow"} {
		r, b, d := residentGraphRunner(t, "ptx-resident-diagadmission-hybrid-v1")
		guardSeen := false
		before := 0
		d.afterDownload = func(v [][]float32) {
			if len(v) != 4 || len(v[3]) != 1 {
				return
			}
			guardSeen = true
			before = d.accumulations
			switch kind {
			case "invalidflag":
				v[3][0] = 1
			case "flagshape":
				v[3] = nil
			case "flagnan":
				v[3][0] = float32(math.NaN())
			case "flagalias":
				v[3] = v[0][:1]
				v[0][0] = 0
			case "hostoverflow":
				v[2][0] = math.MaxFloat32
			}
		}
		state, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
		if _, err := r.Step(context.Background()); err == nil {
			t.Fatal("bad guard accepted", kind)
		}
		if !guardSeen || d.accumulations != before || !reflect.DeepEqual(state, mustFullState(t, r.trainer)) || string(raw) != string(snapshotRunFixture(t, r.run)) {
			t.Fatal("backward/publication after failed admission", kind)
		}
		tapes := d.tapes
		if _, err := r.Step(context.Background()); err == nil || tapes != d.tapes || b.Close() == nil {
			t.Fatal("retry/owner")
		}
	}
}
func TestTrainingResidentDiagonalAdmissionOwnerAndBudget(t *testing.T) {
	for _, kind := range []string{"owner", "logvar", "budget"} {
		r, b, d := residentGraphRunner(t, "ptx-resident-diagadmission-hybrid-v1")
		if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
			t.Fatal(err)
		}
		in := primaryFixture()
		input := DiagonalForwardInputs{Condition: in.Condition, Target: in.Target, Noise: in.Noise, Time: .3}
		m := r.trainer.Flow
		lv := float32(.1)
		switch kind {
		case "owner":
			m = &FlowHeadCPU{}
		case "logvar":
			lv = 81
		case "budget":
			b.MaxTapeBytes = 1
		}
		tapes := d.tapes
		if _, err := b.evaluateDiagonalAdmitted(m, input, lv); err == nil || b.terminal == nil || d.tapes != tapes {
			t.Fatal("guard admission", kind)
		}
	}
	old, b, _ := residentGraphRunner(t, "ptx-resident-embeddingcompact-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), old.trainer); err != nil {
		t.Fatal(err)
	}
	if _, err := b.evaluateDiagonalAdmitted(old.trainer.Flow, DiagonalForwardInputs{}, .1); err == nil || !strings.Contains(err.Error(), "scope") {
		t.Fatal("oldscope guard injected", err)
	}
}
func TestTrainingResidentDiagonalAdmissionAccounting(t *testing.T) {
	old, ob, a := residentGraphRunner(t, "ptx-resident-embeddingcompact-hybrid-v1")
	next, nb, b := residentGraphRunner(t, "ptx-resident-diagadmission-hybrid-v1")
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

func TestTrainingResidentDiagonalAdmissionCrossCompactRestore(t *testing.T) {
	r, b, _ := residentGraphRunner(t, "ptx-resident-diagadmission-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	limits := TrainingBundleLimits{1 << 20, 1 << 16}
	if _, err := r.Checkpoint(dir, "guard", limits); err != nil {
		t.Fatal(err)
	}
	old, ob, _ := residentGraphRunner(t, "ptx-resident-embeddingcompact-hybrid-v1")
	before := mustFullState(t, old.trainer)
	if old.Restore(dir, limits) == nil || !reflect.DeepEqual(before, mustFullState(t, old.trainer)) {
		t.Fatal("crossscope compact restore")
	}
	if b.Close() != nil || ob.Close() != nil {
		t.Fatal("close")
	}
}

func TestTrainingDiagonalAdmissionMissingOutputs(t *testing.T) {
	// Use a complete single constant parameter/gradient inventory and only
	// two boundary positions, rather than failing ordinary source admission.
	p := DiagonalForwardTape{Tape: train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{1}}, {Shape: []int{1}, Values: []float32{0}}}, Outputs: []int{1, 0}}, ResidentBindings: map[string]train.ResidentTapeBinding{"p": {Parameter: 0, Gradient: 1}}}
	if _, err := appendDiagonalAdmission(p, 0, 1024); err == nil || !strings.Contains(err.Error(), "source outputs") {
		t.Fatal("missing boundary", err)
	}
	// The standalone legacy loss ceiling is still enforced even when a compact
	// constant-only forward graph itself would fit the same byte budget.
	p.Tape.Outputs = []int{1, 1, 1}
	if _, err := appendDiagonalAdmission(p, 0, 16); err == nil || !strings.Contains(err.Error(), "loss tape byte ceiling") {
		t.Fatal("loss budget", err)
	}
}
func TestTrainingResidentDiagonalAdmissionBorrowedFlag(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-diagadmission-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	in := DiagonalForwardInputs{Condition: make([]float32, r.trainer.Flow.Condition.In), Target: make([]float32, r.trainer.Flow.Input.In), Noise: make([]float32, r.trainer.Flow.Input.In), Time: .3}
	d.afterDownload = func(values [][]float32) { values[3] = in.Target[:1] }
	before := d.accumulations
	if _, err := b.evaluateDiagonalAdmitted(r.trainer.Flow, in, 0); err == nil || !strings.Contains(err.Error(), "borrowed flag") || d.accumulations != before || b.terminal == nil {
		t.Fatal("borrowed scalar", err)
	}
}
