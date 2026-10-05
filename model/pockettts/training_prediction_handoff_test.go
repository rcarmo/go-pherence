//go:build linux

package pockettts

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"testing"
)

func TestTrainingResidentPredictionHandoffLifecycle(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-predictionhandoff-hybrid-v1")
	childRuns, backwardReads, childCloses := 0, 0, 0
	childOwners := map[*hostResidentGraphTape]bool{}
	d.handoffAfterRun = func(owner *hostResidentGraphTape) {
		for _, c := range owner.tape.Commands {
			if c.Op == "lsd-loss" && c.Mode == 2 {
				childOwners[owner] = true
				childRuns++
				if b.predictionLease != nil {
					t.Fatal("child already live during forward construction")
				}
				return
			}
		}
		if b.predictionLease == nil {
			return
		}
		for _, value := range b.predictionValues {
			if len(value) == 0 {
				t.Fatal("empty roots")
			}
		}
		// Every live producer root is present in an admitted constant binding.
		found := 0
		for _, v := range owner.borrowed {
			for child := range childOwners {
				if child.closed {
					continue
				}
				for _, id := range child.tape.Outputs {
					source := child.values[id]
					if len(source) > 1 && len(source) == len(v) && &source[0] == &v[0] {
						found++
					}
				}
			}
		}
		if found == 0 {
			t.Fatal("backward did not borrow prediction child root")
		}
		backwardReads++
	}
	d.handoffBeforeClose = func(owner *hostResidentGraphTape) error {
		if childOwners[owner] {
			if b.predictionLease == nil || d.accumulations == 0 {
				t.Fatal("child close before final backward/accumulation")
			}
			childCloses++
		}
		return nil
	}
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if childRuns != 4 || backwardReads != 4 || childCloses != 4 || b.predictionLease != nil || b.handoffLease != nil {
		t.Fatal("child lifecycle", childRuns, backwardReads, childCloses)
	}
	if b.Close() != nil {
		t.Fatal("completed close")
	}
}
func TestTrainingResidentPredictionHandoffAccounting(t *testing.T) {
	type cost struct {
		tapes, commands, downloads, elements, accumulations int
		bytes, storage                                      int64
	}
	var costs []cost
	for _, scope := range []string{"ptx-resident-eosadmission-hybrid-v1", "ptx-resident-predictionhandoff-hybrid-v1"} {
		r, _, d := residentGraphRunner(t, scope)
		if _, err := r.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
		costs = append(costs, cost{d.tapes, d.graphCommands, d.downloads, d.downloadedElements, d.accumulations, d.graphBytes, d.storageBytes})
	}
	t.Logf("EOSadmission %+v predictionhandoff %+v", costs[0], costs[1])
	if costs[0].tapes != costs[1].tapes || costs[0].downloads != costs[1].downloads || costs[0].elements != costs[1].elements || costs[0].accumulations != costs[1].accumulations {
		t.Fatal("unexpected download/accumulation change")
	}
}
func predictionDiagonalFixture(t *testing.T) (*TrainingRunner, *ResidentTrainingBackend, *hostResidentGraphDriver, DiagonalForwardInputs, DiagonalForwardResult) {
	t.Helper()
	r, b, d := residentGraphRunner(t, "ptx-resident-predictionhandoff-hybrid-v1")
	o := loadTrainingStepOracle(t)
	batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: o.NormalizedLatents, VoiceLatents: o.VoiceLatents, TextTokens: o.TextTokens}
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	tape, _, err := b.evaluateForwardResults(r.trainer.FlowLM, batch, o.Mask, .1)
	if err != nil {
		t.Fatal(err)
	}
	b.handoffRow = 0
	n, h := r.trainer.FlowLM.LatentDim, r.trainer.FlowLM.Hidden
	in := DiagonalForwardInputs{Condition: tape.output.Z[:h], Target: o.NormalizedLatents[:n], Noise: o.Noise[:n], Time: o.DiagonalTime[0]}
	forward, err := b.evaluateDiagonalAdmitted(r.trainer.Flow, in, .1)
	if err != nil {
		t.Fatal(err)
	}
	return r, b, d, in, forward
}
func TestTrainingResidentPredictionHandoffBackwardAdmission(t *testing.T) {
	for _, kind := range []string{"kind", "row", "borrow", "extent", "parameter", "duplicate", "source", "budget"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d, in, forward := predictionDiagonalFixture(t)
			p, err := r.trainer.Flow.buildDiagonalTrainingTape(in.Condition, in.Time, forward.Input, forward.Prediction, forward.Desired, .1, .5, b.MaxTapeBytes, true)
			if err != nil {
				t.Fatal(err)
			}
			values := [][]float32{forward.Input, forward.Desired, forward.Prediction}
			ids := []int{p.inputID, p.desiredID, p.predictionID}
			switch kind {
			case "kind":
				b.predictionKind = "distill"
			case "row":
				b.predictionRow++
			case "borrow":
				values[0] = append([]float32(nil), values[0]...)
			case "extent":
				ids[0] = p.Tape.Outputs[2]
			case "parameter":
				ids[0] = p.ResidentBindings["flow_lm.flow_net.cond_embed.weight"].Parameter
			case "duplicate":
				ids[1] = ids[0]
			case "source":
				p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "budget":
				var bytes int64
				for _, s := range p.Tape.Tensors {
					n, _ := s.Elements()
					bytes += int64(n) * 4
				}
				b.MaxTapeBytes = bytes
			}
			before := d.tapes
			_, err = b.runPredictionBackward(p.Tape, p.ResidentBindings, p.Tape.Outputs[:3], in.Condition, "diagonal", values, ids, 1)
			if err == nil || d.tapes != before {
				t.Fatal("backward admission allocation", kind, err)
			}
			b.fail(err)
			if b.predictionLease == nil || b.handoffLease == nil || b.predictionLease.Close() == nil || b.Close() == nil {
				t.Fatal("failed child/root freed")
			}
		})
	}
}
func TestTrainingResidentPredictionHandoffChildCloseFailure(t *testing.T) {
	r, b, d := residentGraphRunner(t, "vulkan-resident-predictionhandoff-hybrid-v1")
	sentinel := fmt.Errorf("child close uncertain")
	d.handoffBeforeClose = func(owner *hostResidentGraphTape) error {
		for _, c := range owner.tape.Commands {
			if c.Op == "lsd-loss" && c.Mode == 2 {
				return sentinel
			}
		}
		return nil
	}
	state, raw := mustFullState(t, r.trainer), snapshotRunFixture(t, r.run)
	metrics, err := r.Step(context.Background())
	if err == nil || metrics != (TrainingStepMetrics{}) || d.updates != 0 || b.predictionLease == nil || b.handoffLease == nil || !reflect.DeepEqual(state, mustFullState(t, r.trainer)) || string(raw) != string(snapshotRunFixture(t, r.run)) {
		t.Fatal("child close publication", err)
	}
	if b.Close() == nil || b.closePredictionHandoff() == nil {
		t.Fatal("child freed after close failure")
	}
}
func TestTrainingResidentPredictionHandoffGuardFailure(t *testing.T) {
	for _, kind := range []string{"diagonal", "distill"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-predictionhandoff-hybrid-v1")
			atFailure := -1
			d.afterDownload = func(out [][]float32) {
				if (kind == "diagonal" && len(out) == 4 && len(out[1]) > 1) || (kind == "distill" && len(out) == 7) {
					out[len(out)-1][0] = 1
					atFailure = d.accumulations
				}
			}
			metrics, err := r.Step(context.Background())
			if err == nil || metrics != (TrainingStepMetrics{}) || d.updates != 0 || atFailure < 0 || d.accumulations != atFailure || b.predictionLease == nil || b.handoffLease == nil {
				t.Fatal("forward guard ordering", err, atFailure)
			}
			if b.predictionLease.Close() == nil || b.Close() == nil {
				t.Fatal("guard released child")
			}
		})
	}
}
func TestTrainingResidentPredictionHandoffCrossRestore(t *testing.T) {
	r, _, _ := residentGraphRunner(t, "ptx-resident-predictionhandoff-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	limits := TrainingBundleLimits{1 << 20, 1 << 16}
	dir := t.TempDir()
	if _, err := r.Checkpoint(dir, "child", limits); err != nil {
		t.Fatal(err)
	}
	other, _, _ := residentGraphRunner(t, "ptx-resident-eosadmission-hybrid-v1")
	state := mustFullState(t, other.trainer)
	if other.Restore(dir, limits) == nil || !reflect.DeepEqual(state, mustFullState(t, other.trainer)) {
		t.Fatal("crossscope restore")
	}
}

func TestTrainingResidentPredictionHandoffCancel(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-predictionhandoff-hybrid-v1")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.afterDownload = func(out [][]float32) {
		if b.predictionLease != nil && len(out) == 7 {
			cancel()
		}
	}
	metrics, err := r.Step(ctx)
	if err == nil || metrics != (TrainingStepMetrics{}) || d.updates != 0 || b.predictionLease == nil || b.handoffLease == nil || b.terminal == nil {
		t.Fatal("cancelled child published", err)
	}
	if b.Close() == nil || b.predictionLease.Close() == nil {
		t.Fatal("cancelled child released")
	}
}
func TestTrainingResidentPredictionHandoffFullSourceBudget(t *testing.T) {
	r, b, d, in, forward := predictionDiagonalFixture(t)
	p, err := r.trainer.Flow.buildDiagonalTrainingTape(in.Condition, in.Time, forward.Input, forward.Prediction, forward.Desired, .1, .5, b.MaxTapeBytes, true)
	if err != nil {
		t.Fatal(err)
	}
	before := d.tapes
	// Tight standalone admission must not let pruning hide a late unused
	// constant. All-live combined ceilings are separately tested by the lease.
	var bytes int64
	for _, spec := range p.Tape.Tensors {
		n, _ := spec.Elements()
		bytes += int64(n) * 4
	}
	b.MaxTapeBytes = bytes
	p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{0}})
	_, err = b.runPredictionBackward(p.Tape, p.ResidentBindings, p.Tape.Outputs[:3], in.Condition, "diagonal", [][]float32{forward.Input, forward.Desired, forward.Prediction}, []int{p.inputID, p.desiredID, p.predictionID}, 1)
	if err == nil || d.tapes != before {
		t.Fatal("late fullsource allocated")
	}
	b.fail(err)
	if b.predictionLease.Close() == nil {
		t.Fatal("budget released child")
	}
}
func TestTrainingResidentPredictionHandoffHelpers(t *testing.T) {
	if (*ResidentTrainingBackend)(nil).closePredictionHandoff() != nil {
		t.Fatal("nil close")
	}
	b := &ResidentTrainingBackend{}
	if b.closePredictionHandoff() != nil {
		t.Fatal("empty close")
	}
	if _, err := b.beginPredictionHandoff(train.Tape{}, nil, nil, nil, "diagonal", 3); err == nil {
		t.Fatal("missing owner")
	}
	r, b, _, in, forward := predictionDiagonalFixture(t)
	if _, err := b.beginPredictionHandoff(train.Tape{}, nil, nil, in.Condition, "diagonal", 3); err == nil {
		t.Fatal("reused child")
	}
	p, err := r.trainer.Flow.buildDiagonalTrainingTape(in.Condition, in.Time, forward.Input, forward.Prediction, forward.Desired, .1, .5, b.MaxTapeBytes, true)
	if err != nil {
		t.Fatal(err)
	}
	b.predictionRoots = nil
	if _, err := b.runPredictionBackward(p.Tape, p.ResidentBindings, p.Tape.Outputs[:3], in.Condition, "diagonal", [][]float32{forward.Input, forward.Desired, forward.Prediction}, []int{p.inputID, p.desiredID, p.predictionID}, 1); err == nil {
		t.Fatal("missing root tuple")
	}
	if b.closeFlowHandoff() == nil {
		t.Fatal("live child root close")
	}
}

func TestTrainingResidentPredictionHandoffConstantProvenance(t *testing.T) {
	r, b, d, in, forward := predictionDiagonalFixture(t)
	p, err := r.trainer.Flow.buildDiagonalTrainingTape(in.Condition, in.Time, forward.Input, forward.Prediction, forward.Desired, .1, .5, b.MaxTapeBytes, true)
	if err != nil {
		t.Fatal(err)
	}
	p.Tape.Tensors[p.desiredID].Values[0] += 1
	before := d.tapes
	_, err = b.runPredictionBackward(p.Tape, p.ResidentBindings, p.Tape.Outputs[:3], in.Condition, "diagonal", [][]float32{forward.Input, forward.Desired, forward.Prediction}, []int{p.inputID, p.desiredID, p.predictionID}, 1)
	if err == nil || d.tapes != before {
		t.Fatal("changed constant admitted")
	}
	b.fail(err)
	if b.closePredictionHandoff() == nil {
		t.Fatal("provenance child freed")
	}
}
func TestTrainingResidentPredictionHandoffForwardFailure(t *testing.T) {
	for _, kind := range []string{"construction", "download", "condition"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d, in, _ := predictionDiagonalFixture(t)
			if err := b.closePredictionHandoff(); err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "construction":
				d.failTape = d.tapes + 1
			case "download":
				d.afterDownload = func(out [][]float32) {
					if len(out) == 4 {
						out[0] = nil
					}
				}
			case "condition":
				in.Condition = append([]float32(nil), in.Condition...)
			}
			before := d.accumulations
			if _, err := b.evaluateDiagonalAdmitted(r.trainer.Flow, in, .1); err == nil || d.accumulations != before || b.terminal == nil || b.handoffLease == nil {
				t.Fatal("forward failure", kind)
			}
			if b.Close() == nil {
				t.Fatal("failed producer freed")
			}
		})
	}
}

func TestTrainingExternalTapeIndependentReference(t *testing.T) {
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{3}, External: true}, {Shape: []int{3}, Values: []float32{.2, .3, -.4}}, {Shape: []int{3}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: .7}}, Outputs: []int{2}}
	source := []float32{.1, -.5, .8}
	out := interpretFlowTapeStorageBound(t, tape, true, map[int][]float32{0: source})
	for i, v := range out[0] {
		want := float64(source[i]) + float64(float32(.7))*float64(tape.Tensors[1].Values[i])
		if math.Abs(float64(v)-want) > 2e-6 {
			t.Fatal("external F64 reference")
		}
	}
	if !reflect.DeepEqual(source, []float32{.1, -.5, .8}) {
		t.Fatal("external source mutated")
	}
}
