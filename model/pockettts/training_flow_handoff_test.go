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

// Explicit offline admission; no native loader or execution is used here.
func (d *hostResidentGraphDriver) ValidateHandoffTape(ctx context.Context, t train.Tape, bound map[int]train.ResidentBuffer, maxBytes int64) error {
	if ctx == nil {
		return fmt.Errorf("nil mock handoff context")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := t.Validate(maxBytes); err != nil {
		return err
	}
	if err := train.ValidateExternalTapeBindings(t, bound); err != nil {
		return err
	}
	seen := map[*hostOptimizerBuffer]bool{}
	for id, b := range bound {
		if id < 0 || id >= len(t.Tensors) || (t.Tensors[id].Values == nil && !t.Tensors[id].External) {
			return fmt.Errorf("mock handoff constant")
		}
		x, ok := b.(*hostOptimizerBuffer)
		if !ok || x == nil {
			return fmt.Errorf("mock handoff buffer")
		}
		n, _ := t.Tensors[id].Elements()
		if len(x.values) != n || seen[x] {
			return fmt.Errorf("mock handoff extent/alias")
		}
		seen[x] = true
	}
	return nil
}
func TestTrainingResidentFlowHandoffFullStep(t *testing.T) {
	for _, scope := range []string{"ptx-resident-flowhandoff-hybrid-v1", "vulkan-resident-flowhandoff-hybrid-v1"} {
		t.Run(scope, func(t *testing.T) { testResidentGraphFullStep(t, scope) })
	}
}
func TestTrainingResidentFlowHandoffWeights(t *testing.T) {
	for _, weight := range []float32{0, .25, 1} {
		t.Run(fmt.Sprint(weight), func(t *testing.T) {
			config := DefaultTrainingStepConfig()
			config.PEqual = weight
			testResidentGraphFullStepConfig(t, "ptx-resident-flowhandoff-hybrid-v1", config)
		})
	}
}
func TestTrainingResidentFlowHandoffPublicForward(t *testing.T) {
	oracle := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, oracle)
	trainer, _ := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	b, d := installResidentGraph(t, trainer, "ptx-resident-flowhandoff-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), trainer); err != nil {
		t.Fatal(err)
	}
	in := DistillForwardInputs{Condition: make([]float32, flow.Condition.In), Target: []float32{.4, -.2}, Noise: []float32{-.1, .3}, S: .3, T: .8}
	if _, err := b.EvaluateDistillForward(flow, in); err != nil {
		t.Fatal(err)
	}
	if b.handoffLease != nil || d.accumulations != 0 {
		t.Fatal("public forward created lease/gradient")
	}
}

func TestTrainingResidentFlowHandoffLifecycle(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-flowhandoff-hybrid-v1")
	consumers, producerCloses := 0, 0
	var producer []float32
	d.handoffAfterRun = func(owner *hostResidentGraphTape) {
		for _, c := range owner.tape.Commands {
			if c.Op == "eos" && c.Mode == 2 {
				for _, output := range owner.tape.Outputs {
					if len(owner.values[output]) > 1 {
						producer = owner.values[output]
						break
					}
				}
				return
			}
		}
		if b.handoffLease == nil || b.handoffRow < 0 {
			return
		}
		for _, c := range owner.tape.Commands {
			if c.Op == "vector" && c.Mode == 1 && len(owner.borrowed[c.Inputs[0]]) == len(b.handoffHidden) && c.Offset == b.handoffRow*b.handoffWidth {
				source := owner.borrowed[c.Inputs[0]]
				if producer == nil || &source[0] != &producer[0] {
					t.Fatalf("consumer did not borrow producer device root source=%p producer=%p len=%d outputs=%v row=%d", source, producer, len(source), owner.tape.Outputs, b.handoffRow)
				}
				consumers++
				break
			}
		}
	}
	d.handoffAfterClose = func(owner *hostResidentGraphTape) {
		if producer != nil {
			for _, x := range owner.values {
				if len(x) == len(producer) && len(x) > 0 && &x[0] == &producer[0] {
					// Borrowed consumer storage also contains this root; producer additionally
					// owns the EOS scalar command.
					for _, c := range owner.tape.Commands {
						if c.Op == "eos" && c.Mode == 2 {
							producerCloses++
						}
					}
				}
			}
		}
	}
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if consumers == 0 || producerCloses != 1 || b.handoffLease != nil || b.terminal != nil {
		t.Fatal("lease lifecycle", consumers, producerCloses)
	}
	if b.Close() != nil {
		t.Fatal("completed close")
	}
}
func TestTrainingResidentFlowHandoffAccounting(t *testing.T) {
	type cost struct{ tapes, commands, downloads, elements, storage int }
	var results []cost
	for _, scope := range []string{"ptx-resident-distilladmission-hybrid-v1", "ptx-resident-flowhandoff-hybrid-v1"} {
		r, _, d := residentGraphRunner(t, scope)
		if _, err := r.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
		results = append(results, cost{d.tapes, d.graphCommands, d.downloads, d.downloadedElements, int(d.storageBytes)})
	}
	t.Logf("legacy %+v handoff %+v", results[0], results[1])
	if results[0].tapes != results[1].tapes || results[0].downloads != results[1].downloads || results[0].elements != results[1].elements {
		t.Fatal("unexpected host download claim")
	}
}

func TestTrainingResidentFlowHandoffLateRowFailure(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-flowhandoff-hybrid-v1")
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
	if r.poison == nil || b.terminal == nil || b.state == nil || b.handoffLease == nil {
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

func TestTrainingResidentFlowHandoffRowAdmission(t *testing.T) {
	for _, kind := range []string{"row", "width", "hidden", "provenance", "source", "inventory", "conditionvalue", "latebudget"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-flowhandoff-hybrid-v1")
			oracle := loadTrainingStepOracle(t)
			batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
			samples := TrainingStepSamples{Mask: oracle.Mask, Noise: oracle.Noise, DiagonalTime: oracle.DiagonalTime}
			if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
				t.Fatal(err)
			}
			p, _, err := b.evaluateForwardResults(r.trainer.FlowLM, batch, samples.Mask, .1)
			if err != nil {
				t.Fatal(err)
			}
			b.handoffRow = 0
			condition := p.output.Z[:r.trainer.FlowLM.Hidden]
			in := DiagonalForwardInputs{Condition: condition, Target: batch.NormalizedLatents[:r.trainer.FlowLM.LatentDim], Noise: samples.Noise[:r.trainer.FlowLM.LatentDim], Time: samples.DiagonalTime[0]}
			tape, err := r.trainer.Flow.buildDiagonalForwardTape(in, b.MaxTapeBytes, true)
			if err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "row":
				b.handoffRow = -1
			case "width":
				b.handoffWidth = 0
			case "hidden":
				b.handoffHidden = b.handoffHidden[:1]
			case "provenance":
				condition = append([]float32(nil), condition...)
			case "source":
				tape.Tape.Tensors = append(tape.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "inventory":
				delete(tape.ResidentBindings, "flow_lm.flow_net.cond_embed.weight")
			case "conditionvalue":
				for _, c := range tape.Tape.Commands {
					if c.Op == "linear-forward" && len(c.Inputs) == 4 && c.Inputs[2] == tape.ResidentBindings["flow_lm.flow_net.cond_embed.weight"].Parameter { // Reshape input refers to source constant.
						id := c.Inputs[0]
						for _, reshape := range tape.Tape.Commands {
							if reshape.Op == "reshape" && reshape.Output == id {
								tape.Tape.Tensors[reshape.Inputs[0]].Values[0] += 1
								break
							}
						}
						break
					}
				}
			case "latebudget":
				var bytes int64
				for _, s := range tape.Tape.Tensors {
					n, _ := s.Elements()
					bytes += int64(n) * 4
				}
				b.MaxTapeBytes = bytes
			}
			before := d.tapes
			_, err = b.runFlowHandoffTape(tape.Tape, tape.ResidentBindings, tape.Tape.Outputs[:3], condition)
			if err == nil || d.tapes != before {
				t.Fatal("row admission allocation", kind, err)
			}
			b.fail(err)
			if b.handoffLease == nil || b.closeFlowHandoff() == nil || b.Close() == nil {
				t.Fatal("late owner retention")
			}
		})
	}
}

func TestTrainingResidentFlowHandoffCloseFailure(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-flowhandoff-hybrid-v1")
	sentinel := fmt.Errorf("producer close uncertain")
	closed := false
	d.handoffBeforeClose = func(owner *hostResidentGraphTape) error {
		for _, command := range owner.tape.Commands {
			if command.Op == "eos" && command.Mode == 2 {
				closed = true
				return sentinel
			}
		}
		return nil
	}
	state := mustFullState(t, r.trainer)
	raw := snapshotRunFixture(t, r.run)
	metrics, err := r.Step(context.Background())
	if err == nil || !closed || b.handoffLease == nil || metrics != (TrainingStepMetrics{}) || d.updates != 0 || !reflect.DeepEqual(state, mustFullState(t, r.trainer)) || string(raw) != string(snapshotRunFixture(t, r.run)) {
		t.Fatal("close failure publication", err)
	}
	if b.Close() == nil || b.closeFlowHandoff() == nil {
		t.Fatal("close failure freed producer")
	}
}
func TestTrainingResidentFlowHandoffCancel(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-flowhandoff-hybrid-v1")
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	d.afterDownload = func(values [][]float32) {
		if b.handoffLease != nil && b.handoffRow == 0 && len(values) == 7 {
			cancel()
		}
	}
	metrics, err := r.Step(ctx)
	if err == nil || metrics != (TrainingStepMetrics{}) || b.handoffLease == nil || b.terminal == nil || d.updates != 0 {
		t.Fatal("cancelled boundary publication", err)
	}
	if b.Close() == nil {
		t.Fatal("cancelled producer freed")
	}
}
func TestTrainingResidentFlowHandoffHelpers(t *testing.T) {
	if b, ok := flowBackendForHandoff(nil); b != nil || ok {
		t.Fatal("nil flow")
	}
	if _, ok := flowBackendForHandoff(&FlowHeadCPU{}); ok {
		t.Fatal("nonresident helper")
	}
	if (*ResidentTrainingBackend)(nil).closeFlowHandoff() != nil {
		t.Fatal("nil close")
	}
	b := &ResidentTrainingBackend{Scope: "ptx-resident-flowhandoff-hybrid-v1"}
	if b.closeFlowHandoff() != nil {
		t.Fatal("empty lease close")
	}
}
func TestTrainingResidentFlowHandoffCrossRestore(t *testing.T) {
	r, _, _ := residentGraphRunner(t, "ptx-resident-flowhandoff-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	limits := TrainingBundleLimits{1 << 20, 1 << 16}
	dir := t.TempDir()
	_, err := r.Checkpoint(dir, "handoff", limits)
	if err != nil {
		t.Fatal(err)
	}
	other, _, _ := residentGraphRunner(t, "ptx-resident-distilladmission-hybrid-v1")
	before := mustFullState(t, other.trainer)
	if err = other.Restore(dir, limits); err == nil || !reflect.DeepEqual(before, mustFullState(t, other.trainer)) {
		t.Fatal("crossscope restore")
	}
}

func TestTrainingEOSAdmissionTapeReference(t *testing.T) {
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{4}, Values: []float32{.1, .2, .3, .4}}, {Shape: []int{2}, Values: []float32{.5, .6}}, {Shape: []int{1}, Values: []float32{.1}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "eos-admission", Inputs: []int{0, 1, 2}, Output: 3, Rows: 2, Width: 2}}, Outputs: []int{3}}
	values := interpretFlowTape(t, tape)
	if len(values) != 1 || values[0][0] != 0 {
		t.Fatal("scalar admission interpreter")
	}
	compact, mapping, err := train.PruneTape(tape, []int{3}, 1024)
	if err != nil || mapping[3] < 0 || len(compact.Commands) != 1 {
		t.Fatal("pruned scalar admission", err)
	}
}
