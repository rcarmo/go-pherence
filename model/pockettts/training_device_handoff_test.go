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

func TestTrainingResidentDeviceHandoffFullStep(t *testing.T) {
	for _, scope := range []string{"ptx-resident-devicehandoff-hybrid-v1", "vulkan-resident-devicehandoff-hybrid-v1"} {
		t.Run(scope, func(t *testing.T) { testResidentGraphFullStep(t, scope) })
	}
}
func TestTrainingResidentDeviceHandoffWeights(t *testing.T) {
	for _, weight := range []float32{0, .25, 1} {
		t.Run(fmt.Sprint(weight), func(t *testing.T) {
			config := DefaultTrainingStepConfig()
			config.PEqual = weight
			testResidentGraphFullStepConfig(t, "ptx-resident-devicehandoff-hybrid-v1", config)
		})
	}
}
func TestTrainingResidentDeviceHandoffLifecycle(t *testing.T) {
	r, b, d := residentGraphRunner(t, "ptx-resident-devicehandoff-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	if b.deviceCondition != nil || b.deviceDiagonal != nil || b.deviceDistill != nil || b.handoffLease != nil || b.predictionLease != nil || b.deviceValues != nil {
		t.Fatal("retained row state after fullstep")
	}
	t.Logf("device scope accounting tapes%d commands%d downloads%d elements%d accumulations%d storage%d", d.tapes, d.graphCommands, d.downloads, d.downloadedElements, d.accumulations, d.storageBytes)
}
func TestTrainingResidentDeviceHandoffFailure(t *testing.T) {
	r, _, d := residentGraphRunner(t, "ptx-resident-devicehandoff-hybrid-v1")
	if _, err := r.Step(context.Background()); err != nil {
		t.Fatal(err)
	}
	count := d.tapes
	for fail := 1; fail <= count; fail++ {
		t.Run(fmt.Sprint(fail), func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-devicehandoff-hybrid-v1")
			before, err := r.trainer.State()
			if err != nil {
				t.Fatal(err)
			}
			d.failTape = fail
			if _, err := r.Step(context.Background()); err == nil || b.terminal == nil {
				t.Fatal("failed tape accepted")
			}
			after, _ := r.trainer.State()
			if !reflect.DeepEqual(before, after) {
				t.Fatal("failed update published")
			}
			count := d.tapes
			if _, err := r.Step(context.Background()); err == nil || d.tapes != count {
				t.Fatal("failed retry")
			}
			if b.Close() == nil {
				t.Fatal("uncertain release")
			}
		})
	}
}
func TestTrainingResidentDeviceHandoffScopes(t *testing.T) {
	for _, scope := range []string{"ptx-resident-devicehandoff-hybrid-v1", "vulkan-resident-devicehandoff-hybrid-v1"} {
		b := &ResidentTrainingBackend{Scope: scope}
		if !b.deviceHandoffEnabled() || !b.eosAdmissionEnabled() || !b.flowHandoffEnabled() || !b.diagonalAdmissionEnabled() || !b.distillAdmissionEnabled() || !b.distillResultsEnabled() || !b.ownsWeight() {
			t.Fatal("inheritance", scope)
		}
	}
	legacy := &ResidentTrainingBackend{Scope: "ptx-resident-predictionhandoff-hybrid-v1"}
	if legacy.deviceHandoffEnabled() {
		t.Fatal("legacy route changed")
	}
	if !reflect.DeepEqual(legacy.deviceValues, [][]float32(nil)) {
		t.Fatal("fixture")
	}
}

func TestTrainingResidentDeviceHandoffProvenance(t *testing.T) {
	for _, kind := range []string{"kind", "row", "copy", "changed", "condition", "missing"} {
		t.Run(kind, func(t *testing.T) {
			hidden := []float32{.2, -.3, .5}
			v := []float32{.1, .2}
			b := &ResidentTrainingBackend{deviceCondition: &train.ResidentTapeLease{}, deviceKind: "distill", deviceRow: 0, handoffRow: 0, handoffFrames: 1, handoffWidth: 3, handoffHidden: hidden}
			b.rememberDeviceValues("distill", [][]float32{v})
			condition := hidden
			values := [][]float32{v}
			switch kind {
			case "kind":
				b.deviceKind = "wrong"
			case "row":
				b.deviceRow = -1
			case "copy":
				values[0] = append([]float32(nil), v...)
			case "changed":
				v[0] += 1
			case "condition":
				condition = append([]float32(nil), hidden...)
			case "missing":
				b.deviceCondition = nil
			}
			if b.validateDeviceValues("distill", condition, values) == nil {
				t.Fatal("provenance accepted", kind)
			}
		})
	}
}

func deviceHandoffRowFixture(t *testing.T) (*FlowHeadCPU, *ResidentTrainingBackend, *hostResidentGraphDriver, []float32) {
	t.Helper()
	o := loadTrainingStepOracle(t)
	lm, flow, w := trainingStepModelsFromOracle(t, o)
	trainer, err := NewFullTrainer(lm, flow, w, DefaultAdamWConfig(), .9)
	if err != nil {
		t.Fatal(err)
	}
	b, d := installResidentGraph(t, trainer, "ptx-resident-devicehandoff-hybrid-v1")
	if err = b.PrepareTrainingStep(context.Background(), trainer); err != nil {
		t.Fatal(err)
	}
	batch := FlowLMTrainingBatch{Frames: o.Frames, VoiceFrames: o.VoiceFrames, NormalizedLatents: o.NormalizedLatents, VoiceLatents: o.VoiceLatents, TextTokens: o.TextTokens}
	f, _, err := b.evaluateForwardResults(lm, batch, o.Mask, DefaultTrainingStepConfig().EOSLossWeight)
	if err != nil {
		t.Fatal(err)
	}
	b.handoffRow = 0
	return flow, b, d, f.output.Z[:flow.Condition.In]
}
func TestTrainingResidentDeviceHandoffBackwardAdmission(t *testing.T) {
	for _, kind := range []string{"missing", "kind", "row", "copy", "changed", "condition", "time", "budget"} {
		t.Run("diagonal-"+kind, func(t *testing.T) {
			flow, b, d, condition := deviceHandoffRowFixture(t)
			forward := DiagonalForwardInputs{Condition: condition, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, Time: .3}
			f, err := b.evaluateDiagonalAdmitted(flow, forward, 0)
			if err != nil {
				t.Fatal(err)
			}
			time := float32(.3)
			switch kind {
			case "missing":
				b.deviceDiagonal = nil
			case "kind":
				b.deviceKind = "wrong"
			case "row":
				b.deviceRow = -1
			case "copy":
				f.Input = append([]float32(nil), f.Input...)
			case "changed":
				f.Desired[0] += 1
			case "condition":
				condition = append([]float32(nil), condition...)
			case "time":
				time = .4
			case "budget":
				b.MaxTapeBytes = 16
			}
			before, count := d.tapes, d.accumulations
			if _, err := b.EvaluateDiagonal(flow, condition, time, f.Input, f.Prediction, f.Desired, 0, .25); err == nil || d.tapes != before || d.accumulations != count || b.terminal == nil {
				t.Fatal("diagonal admission crossed device boundary")
			}
			if b.closeFlowHandoff() == nil || b.Close() == nil {
				t.Fatal("failed owner release")
			}
		})
	}
	for _, kind := range []string{"missing", "kind", "target", "noise", "time", "scale", "provenance"} {
		t.Run("distill-"+kind, func(t *testing.T) {
			flow, b, d, condition := deviceHandoffRowFixture(t)
			forward := DistillForwardInputs{Condition: condition, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, S: .3, T: .8}
			f, err := b.evaluateDistillAdmitted(flow, forward, 0)
			if err != nil {
				t.Fatal(err)
			}
			in := DistillTrainingInputs{Condition: condition, PrimaryInput: f.PrimaryInput, Velocity: f.Velocity, Tangent: f.Tangent, EndpointInput: f.EndpointInput, Desired: f.Desired, Prediction: f.Prediction, Target: forward.Target, Noise: forward.Noise, S: .3, T: .8}
			scale := float32(.25)
			switch kind {
			case "missing":
				b.deviceDistill = nil
			case "kind":
				b.deviceKind = "diagonal"
			case "target":
				in.Target = []float32{1, 1}
			case "noise":
				in.Noise = []float32{1, 1}
			case "time":
				in.S = .4
			case "scale":
				scale = -1
			case "provenance":
				b.deviceDistill.provenance = nil
			}
			before, count := d.tapes, d.accumulations
			if _, err := b.evaluateDistillResults(flow, in, scale); err == nil || d.tapes != before || d.accumulations != count || b.terminal == nil {
				t.Fatal("distill admission crossed device boundary")
			}
		})
	}
}
func TestTrainingResidentDeviceHandoffLatePublication(t *testing.T) {
	for _, kind := range []string{"flag", "cancel", "childclose", "conditionclose"} {
		t.Run(kind, func(t *testing.T) {
			r, b, d := residentGraphRunner(t, "ptx-resident-devicehandoff-hybrid-v1")
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			before, err := r.trainer.State()
			if err != nil {
				t.Fatal(err)
			}
			runBefore := r.run.core.clone()
			guards := 0
			injected := false
			d.afterDownload = func(values [][]float32) {
				if len(values) == 2 {
					guards++
					if guards == 3 {
						injected = true
						if kind == "flag" {
							values[0][0] = 1
						}
						if kind == "cancel" {
							cancel()
						}
					}
				}
			}
			d.handoffBeforeClose = func(owner *hostResidentGraphTape) error {
				if kind == "childclose" && len(owner.tape.Outputs) == 6 {
					injected = true
					return fmt.Errorf("injected childclose")
				}
				if kind == "conditionclose" && b.deviceCondition != nil && len(owner.tape.Outputs) == 1 && len(owner.tape.Commands) == 1 {
					injected = true
					return fmt.Errorf("injected rowcondition close")
				}
				return nil
			}
			metrics, err := r.Step(ctx)
			if err == nil || !injected || b.terminal == nil || d.accumulations == 0 || !reflect.DeepEqual(metrics, TrainingStepMetrics{}) {
				t.Fatal("late failure publication", kind, err, guards, d.accumulations)
			}
			after, _ := r.trainer.State()
			if !reflect.DeepEqual(before, after) || !reflect.DeepEqual(runBefore, r.run.core.clone()) {
				t.Fatal("late failure update/replay")
			}
			if _, e := r.Checkpoint(t.TempDir(), "held", TrainingBundleLimits{1 << 20, 1 << 16}); e == nil {
				t.Fatal("failed checkpoint")
			}
			if _, e := b.state.Snapshot(context.Background()); e == nil {
				t.Fatal("failed device snapshot")
			}
			count := d.tapes
			if _, e := r.Step(context.Background()); e == nil || d.tapes != count {
				t.Fatal("retry")
			}
			if b.Close() == nil {
				t.Fatal("uncertainrelease")
			}
		})
	}
}
func TestTrainingResidentDeviceHandoffConditionAdmission(t *testing.T) {
	for _, kind := range []string{"active", "row", "shape", "owner", "copy", "finish-missing"} {
		t.Run(kind, func(t *testing.T) {
			flow, b, d, c := deviceHandoffRowFixture(t)
			_ = flow
			before := d.tapes
			switch kind {
			case "active":
				b.deviceDiagonal = &diagonalDeviceSources{}
			case "row":
				b.handoffRow = -1
			case "shape":
				c = c[:1]
			case "owner":
				b.handoffLease = nil
			case "copy":
				c = append([]float32(nil), c...)
			case "finish-missing":
				if b.finishDeviceHandoff() == nil {
					t.Fatal("missingclose")
				}
				return
			}
			if _, err := b.beginDeviceCondition(c); err == nil || d.tapes != before {
				t.Fatal("row condition guard")
			}
		})
	}
}

func TestTrainingResidentDeviceHandoffForwardFailures(t *testing.T) {
	for _, branch := range []string{"diagonal", "distill"} {
		for _, kind := range []string{"shape", "nan", "alias", "overflow", "download-cancel"} {
			t.Run(branch+"-"+kind, func(t *testing.T) {
				flow, b, d, c := deviceHandoffRowFixture(t)
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				b.ctx = ctx
				d.afterDownload = func(values [][]float32) {
					if branch == "diagonal" && len(values) != 3 || branch == "distill" && len(values) != 6 {
						return
					}
					switch kind {
					case "shape":
						values[0] = nil
					case "nan":
						values[1][0] = float32(math.NaN())
					case "alias":
						values[1] = values[0]
					case "overflow":
						values[len(values)-1][0] = 1e30
					case "download-cancel":
						cancel()
						values[0] = nil
					}
				}
				var err error
				if branch == "diagonal" {
					_, err = b.evaluateDiagonalAdmitted(flow, DiagonalForwardInputs{Condition: c, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, Time: .3}, 0)
				} else {
					_, err = b.evaluateDistillAdmitted(flow, DistillForwardInputs{Condition: c, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, S: .3, T: .8}, 0)
				}
				if err == nil || b.terminal == nil || d.accumulations != 0 {
					t.Fatal("forward failure accumulated/published", err, d.accumulations)
				}
				if b.closeFlowHandoff() == nil {
					t.Fatal("failedtree released")
				}
			})
		}
	}
}
func TestTrainingResidentDeviceHandoffBackwardResults(t *testing.T) {
	for _, branch := range []string{"diagonal", "distill"} {
		t.Run(branch, func(t *testing.T) {
			flow, b, d, c := deviceHandoffRowFixture(t)
			// Return a graph-owned result slice instead of an owned download. The
			// private runner only admits result-result disjointness; integration must
			// also reject overlap with retained host forward/condition arrays.
			if branch == "diagonal" {
				f, err := b.evaluateDiagonalAdmitted(flow, DiagonalForwardInputs{Condition: c, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, Time: .3}, 0)
				if err != nil {
					t.Fatal(err)
				}
				d.afterDownload = func(v [][]float32) {
					if len(v) == 3 {
						v[0] = c
					}
				}
				if _, err := b.EvaluateDiagonal(flow, c, .3, f.Input, f.Prediction, f.Desired, 0, .25); err == nil || b.terminal == nil {
					t.Fatal("borrowed condition result")
				}
			} else {
				f, err := b.evaluateDistillAdmitted(flow, DistillForwardInputs{Condition: c, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, S: .3, T: .8}, 0)
				if err != nil {
					t.Fatal(err)
				}
				d.afterDownload = func(v [][]float32) {
					if len(v) == 5 {
						v[0] = c
					}
				}
				in := DistillTrainingInputs{Condition: c, PrimaryInput: f.PrimaryInput, Velocity: f.Velocity, Tangent: f.Tangent, EndpointInput: f.EndpointInput, Desired: f.Desired, Prediction: f.Prediction, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, S: .3, T: .8}
				if _, err := b.evaluateDistillResults(flow, in, .25); err == nil || b.terminal == nil {
					t.Fatal("borrowed condition result")
				}
			}
		})
	}
}

func TestTrainingResidentDeviceHandoffAccounting(t *testing.T) {
	for _, scope := range []string{"ptx-resident-predictionhandoff-hybrid-v1", "ptx-resident-devicehandoff-hybrid-v1"} {
		r, _, d := residentGraphRunner(t, scope)
		if _, err := r.Step(context.Background()); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s tapes%d commands%d downloads%d elements%d accumulations%d graphbytes%d plannedstorage%d", scope, d.tapes, d.graphCommands, d.downloads, d.downloadedElements, d.accumulations, d.graphBytes, d.storageBytes)
	}
}

func TestImmediateChildExternalInputs(t *testing.T) {
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}}, Commands: []train.Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
	out, err := externalizeImmediateChild(tape, []int{0}, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if out.Tensors[0].Values != nil || !out.Tensors[0].External || tape.Tensors[0].External || tape.Tensors[0].Values == nil {
		t.Fatal("source declaration ownership")
	}
	if &out.Tensors[1].Values[0] != &tape.Tensors[1].Values[0] {
		t.Fatal("immediate-use constant copied")
	}
	for _, ids := range [][]int{nil, {0, 0}, {-1}, {2}, {3}} {
		if _, err := externalizeImmediateChild(tape, ids, 1024); err == nil {
			t.Fatal("invalid selection admitted", ids)
		}
	}
	tape.Tensors[1].Values[0] = float32(math.NaN())
	if _, err := externalizeImmediateChild(tape, []int{0}, 1024); err == nil {
		t.Fatal("complete-source nonfinite admission")
	}
}
