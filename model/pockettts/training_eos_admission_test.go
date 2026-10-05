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

func eosAdmissionProducerFixture(t *testing.T) (BackboneTrainingTape, *FlowLMTrainingCPU, FlowLMTrainingBatch) {
	t.Helper()
	_, m := backboneOracle(t)
	batch := eosForwardBatch(2, 1, []uint32{1, 1})
	p, err := m.buildEOSForwardTrainingTape(batch, []bool{true, false}, .1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	p, err = buildForwardResultsTape(p, m, batch, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	return p, m, batch
}
func TestTrainingResidentEOSAdmissionProducer(t *testing.T) {
	p, m, batch := eosAdmissionProducerFixture(t)
	q, err := appendForwardEOSAdmission(p, m, batch, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if len(q.Tape.Commands) != len(p.Tape.Commands)+2 || len(q.Tape.Tensors) != len(p.Tape.Tensors)+2 || len(q.Tape.Outputs) != len(p.Tape.Outputs)+1 || !reflect.DeepEqual(q.ResidentBindings, p.ResidentBindings) {
		t.Fatal("source contract")
	}
	ids := append(forwardResultsOutputs(p), q.Tape.Outputs[len(q.Tape.Outputs)-1])
	compact, mapping, err := train.PruneTape(q.Tape, ids, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	out := interpretFlowTape(t, compact)
	if len(out) != 4 || out[3][0] != 0 || mapping[ids[0]] < 0 {
		t.Fatal("pruned flag/root")
	}
	original := interpretFlowTape(t, p.Tape)
	expanded := interpretFlowTape(t, q.Tape)
	for i, x := range original {
		assertSliceClose(t, "unchanged producer", expanded[i], x, 2e-6)
	}
	var bytes int64
	for _, s := range q.Tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err := appendForwardEOSAdmission(p, m, batch, bytes-1); err == nil {
		t.Fatal("expanded ceiling")
	}
	if _, err := appendForwardEOSAdmission(p, m, batch, bytes); err != nil {
		t.Fatal(err)
	}
	if len(p.Tape.Tensors)+2 != len(q.Tape.Tensors) {
		t.Fatal("caller graph modified")
	}
}
func TestTrainingResidentEOSAdmissionProducerRejects(t *testing.T) {
	for _, kind := range []string{"source", "topology", "rows", "hidden", "weight", "bias", "extent", "alias", "affine", "selected", "offset"} {
		t.Run(kind, func(t *testing.T) {
			p, m, batch := eosAdmissionProducerFixture(t)
			switch kind {
			case "source":
				p.Tape.Tensors = append(p.Tape.Tensors, train.TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
			case "topology":
				m = nil
			case "rows":
				batch.Frames = 0
			case "hidden":
				batch.Frames = 1
			case "weight":
				delete(p.ResidentBindings, "flow_lm.out_eos.weight")
			case "bias":
				delete(p.ResidentBindings, "flow_lm.out_eos.bias")
			case "extent":
				m.Hidden++
				m.EOS.In++
			case "alias":
				p.ResidentBindings["flow_lm.out_eos.bias"] = p.ResidentBindings["flow_lm.out_eos.weight"]
			case "affine":
				for i, c := range p.Tape.Commands {
					if c.Op == "linear-forward" && c.Inputs[2] == p.ResidentBindings["flow_lm.out_eos.weight"].Parameter {
						p.Tape.Commands[i].Mode = 1
						break
					}
				}
			case "selected":
				n := len(p.Tape.Outputs)
				p.Tape.Outputs[n-2] = p.Tape.Outputs[0]
			case "offset":
				hidden := forwardResultsOutputs(p)[0]
				for i, c := range p.Tape.Commands {
					if c.Output == hidden {
						p.Tape.Commands[i].Offset = 0
					}
				}
			}
			if _, err := appendForwardEOSAdmission(p, m, batch, 64<<20); err == nil {
				t.Fatal("invalid producer", kind)
			}
		})
	}
}
func TestTrainingResidentEOSAdmissionFlagRejectOrdering(t *testing.T) {
	for _, kind := range []string{"flag", "shape", "nan", "alias", "borrowed", "hostoverflow"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-eosadmission-hybrid-v1")
			seen := 0
			d.afterDownload = func(out [][]float32) {
				if len(out) != 4 || len(out[1]) != 1 || len(out[2]) != 1 {
					return
				}
				seen++
				switch kind {
				case "flag":
					out[3][0] = 1
				case "shape":
					out[3] = nil
				case "nan":
					out[3][0] = float32(math.NaN())
				case "alias":
					out[3] = out[2]
				case "borrowed": // Direct validator covers source overlap below.
					out[3] = out[0][:1]
					out[3][0] = 0
				case "hostoverflow":
					for i := range out[0] {
						out[0][i] = math.MaxFloat32
					}
					out[3][0] = 0
				}
			}
			before := mustFullState(t, r.trainer)
			raw := snapshotRunFixture(t, r.run)
			metrics, err := r.Step(context.Background())
			if err == nil || seen != 1 || metrics != (TrainingStepMetrics{}) || d.accumulations != 0 || d.updates != 0 || b.handoffLease == nil || b.terminal == nil || !reflect.DeepEqual(before, mustFullState(t, r.trainer)) || string(raw) != string(snapshotRunFixture(t, r.run)) {
				t.Fatal("flag published gradients/boundary", err, seen, d.accumulations)
			}
			if _, err := r.Checkpoint(t.TempDir(), "failed", TrainingBundleLimits{1 << 20, 1 << 16}); err == nil {
				t.Fatal("checkpoint")
			}
			if _, err := b.state.Snapshot(context.Background()); err == nil {
				t.Fatal("snapshot")
			}
			tapes := d.tapes
			if _, err := r.Step(context.Background()); err == nil || d.tapes != tapes || b.Close() == nil {
				t.Fatal("retry/close")
			}
		})
	}
}
func TestTrainingResidentEOSAdmissionFlagValidation(t *testing.T) {
	batch := FlowLMTrainingBatch{NormalizedLatents: []float32{0, 1}, VoiceLatents: []float32{0}}
	if validateForwardEOSAdmission([][]float32{{1}, {2}, {0}, {0}}, batch) != nil {
		t.Fatal("valid flag")
	}
	if validateForwardEOSAdmission([][]float32{{1}, {2}, {0}, batch.NormalizedLatents[:1]}, batch) == nil {
		t.Fatal("borrowed flag")
	}
}
func TestTrainingResidentEOSAdmissionAccounting(t *testing.T) {
	type cost struct {
		tapes, commands, downloads, elements int
		bytes, storage                       int64
	}
	var costs []cost
	for _, scope := range []string{"ptx-resident-flowhandoff-hybrid-v1", "ptx-resident-eosadmission-hybrid-v1"} {
		r, _, d := residentGraphRunner(t, scope)
		if _, err := r.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
		costs = append(costs, cost{d.tapes, d.graphCommands, d.downloads, d.downloadedElements, d.graphBytes, d.storageBytes})
	}
	t.Logf("handoff %+v EOSadmission %+v", costs[0], costs[1])
	if costs[1].tapes != costs[0].tapes || costs[1].commands != costs[0].commands+2 || costs[1].downloads != costs[0].downloads+1 || costs[1].elements != costs[0].elements+1 {
		t.Fatal("accounting")
	}
}

// Hide optional capabilities without modifying the installed mock driver.
type eosAdmissionNoHandoffDriver struct{ train.ResidentDriver }

func TestTrainingResidentEOSAdmissionCapability(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-eosadmission-hybrid-v1")
	b.Driver = &eosAdmissionNoHandoffDriver{ResidentDriver: d}
	if b.PrepareTrainingStep(context.Background(), r.trainer) == nil || b.state != nil || d.allocations != 0 {
		t.Fatal("missing handoff capability allocated state")
	}
}
func TestTrainingResidentEOSAdmissionConservativePolicy(t *testing.T) {
	// The private appended graph must preserve the primitive's documented
	// rejection of an original-host-safe near-limit bias (including padding).
	p, m, batch := eosAdmissionProducerFixture(t)
	budgetLimit := float64(math.MaxFloat32) * .125 / float64(batch.Frames)
	value := float32(budgetLimit)
	if float64(value) > budgetLimit {
		value = math.Nextafter32(value, 0)
	}
	bias := p.ResidentBindings["flow_lm.out_eos.bias"].Parameter
	weight := p.ResidentBindings["flow_lm.out_eos.weight"].Parameter
	// Own the modified fixture arrays; do not mutate borrowed model parameters.
	p.Tape.Tensors = append([]train.TensorSpec(nil), p.Tape.Tensors...)
	p.Tape.Tensors[bias].Values = []float32{value}
	p.Tape.Tensors[weight].Values = make([]float32, m.Hidden)
	q, err := appendForwardEOSAdmission(p, m, batch, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	out := interpretFlowTape(t, q.Tape)
	if out[len(out)-1][0] != 1 {
		t.Fatal("conservative scope lost rejection")
	}
	if float64(value) > budgetLimit {
		t.Fatal("fixture not host-safe")
	}
}
func TestTrainingResidentEOSAdmissionProducerLifetime(t *testing.T) {
	r, b, d := residentGraphRunner(t, "vulkan-resident-eosadmission-hybrid-v1")
	checked, closed := 0, 0
	d.handoffAfterRun = func(owner *hostResidentGraphTape) {
		for _, command := range owner.tape.Commands {
			if command.Op == "eos-admission" {
				checked++
				if d.accumulations != 0 || owner.closed {
					t.Fatal("guard follows gradients/cleanup")
				}
			}
		}
	}
	d.handoffAfterClose = func(owner *hostResidentGraphTape) {
		for _, command := range owner.tape.Commands {
			if command.Op == "eos-admission" {
				closed++
				if d.accumulations == 0 {
					t.Fatal("producer closed before row consumers")
				}
			}
		}
	}
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if checked != 1 || closed != 1 || b.handoffLease != nil {
		t.Fatal("producer lifecycle", checked, closed)
	}
	if b.Close() != nil {
		t.Fatal("completed close")
	}
}

func TestTrainingResidentEOSAdmissionInheritanceAndRetention(t *testing.T) {
	for _, scope := range []string{"ptx-resident-eosadmission-hybrid-v1", "vulkan-resident-eosadmission-hybrid-v1"} {
		b := &ResidentTrainingBackend{Scope: scope}
		if !b.eosAdmissionEnabled() || !b.flowHandoffEnabled() || !b.distillAdmissionEnabled() || !b.diagonalAdmissionEnabled() || !b.embeddingCompactEnabled() || !b.forwardResultsEnabled() {
			t.Fatal("scope inheritance", scope)
		}
	}
	r, b, _ := residentGraphRunner(t, "ptx-resident-eosadmission-hybrid-v1")
	if err := b.PrepareTrainingStep(context.Background(), r.trainer); err != nil {
		t.Fatal(err)
	}
	oracle := loadTrainingStepOracle(t)
	batch := FlowLMTrainingBatch{Frames: oracle.Frames, VoiceFrames: oracle.VoiceFrames, NormalizedLatents: oracle.NormalizedLatents, VoiceLatents: oracle.VoiceLatents, TextTokens: oracle.TextTokens}
	if _, _, err := b.evaluateForwardResults(r.trainer.FlowLM, batch, oracle.Mask, .1); err != nil {
		t.Fatal(err)
	}
	sentinel := fmt.Errorf("boundary failed")
	if b.fail(sentinel) != sentinel || b.handoffLease == nil || b.handoffLease.Close() == nil {
		t.Fatal("lease not retained")
	}
	b.fail(fmt.Errorf("later"))
	if b.terminal != sentinel {
		t.Fatal("first error lost")
	}
	if _, err := b.state.Snapshot(context.Background()); err == nil || b.Close() == nil {
		t.Fatal("terminal publication/close")
	}
}
