//go:build linux

package pockettts

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func diagonalDeviceFixture(t *testing.T, time float32) (*FlowHeadCPU, *train.ResidentState, *hostResidentGraphDriver, *train.ResidentTapeLease, DiagonalForwardInputs) {
	t.Helper()
	ctx := context.Background()
	m := tinyTrainableFlowHead()
	in := DiagonalForwardInputs{Condition: []float32{.2, -.3, .5}, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, Time: time}
	graph, err := m.buildDiagonalDeviceTrainingTape(3, 2, time, 0, 1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	d := &hostResidentGraphDriver{t: t}
	snapshot := train.ResidentSnapshot{Config: train.ResidentConfig{LearningRate: .001, Beta1: .9, Beta2: .99, Epsilon: 1e-8, EMADecay: .9}, Parameters: map[string]train.ResidentParameter{}}
	for name, b := range graph.bindings {
		p := graph.tape.Tensors[b.Parameter].Values
		snapshot.Parameters[name] = train.ResidentParameter{Parameter: p, M: make([]float32, len(p)), V: make([]float32, len(p)), EMA: append([]float32(nil), p...)}
	}
	s, err := train.NewResidentState(ctx, d, snapshot, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	producer := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{3}, Values: in.Condition}}, Outputs: []int{0}}
	root, err := s.BeginZeroPrunedTapeLease(ctx, producer, nil, producer.Outputs, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	return m, s, d, root, in
}
func TestTrainingDiagonalDeviceInputsParity(t *testing.T) {
	for _, time := range []float32{0, .3, 1} {
		for _, scale := range []float32{0, .25, 1} {
			for _, logvar := range []float32{-2, 0, 1} {
				t.Run(fmt.Sprint(time, scale, logvar), func(t *testing.T) {
					ctx := context.Background()
					m, state, d, root, in := diagonalDeviceFixture(t, time)
					f, err := m.BuildDiagonalForwardTape(in, 64<<20)
					if err != nil {
						t.Fatal(err)
					}
					expectedForward := interpretFlowTape(t, f.Tape)
					legacy, err := m.BuildDiagonalTrainingTape(in.Condition, time, expectedForward[0], expectedForward[2], expectedForward[1], logvar, scale, 64<<20)
					if err != nil {
						t.Fatal(err)
					}
					expected := interpretFlowTape(t, legacy.Tape)
					sources, err := beginDiagonalDeviceSources(ctx, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20)
					if err != nil {
						t.Fatal(err)
					}
					if d.downloadCalls != 0 {
						t.Fatal("forward vector downloaded")
					}
					got, err := runDiagonalDeviceTraining(ctx, m, sources, 0, 3, 2, time, logvar, scale, 64<<20)
					if err != nil {
						t.Fatal(err)
					}
					assertSliceClose(t, "condition", got.DCondition, expected[0], 2e-5)
					assertSliceClose(t, "target", got.DTarget, expected[1], 2e-5)
					assertSliceClose(t, "stats", []float32{float32(got.RawSquare), float32(got.Loss), got.DLogVariance}, expected[2], 2e-6)
					// Independent CPU VJP/loss boundary, same existing tolerance.
					loss, e := BuildLSDLossTrainingTape(expectedForward[2], expectedForward[1], logvar, 64<<20)
					if e != nil {
						t.Fatal(e)
					}
					lossValues := interpretFlowTape(t, loss)
					gv := append([]float32(nil), lossValues[1]...)
					for i := range gv {
						gv[i] *= scale
					}
					_, grad, cond, _, inputGrad, e := m.ForwardBackward(in.Condition, []float32{time, time}, expectedForward[0], gv)
					if e != nil {
						t.Fatal(e)
					}
					assertSliceClose(t, "CPU condition", got.DCondition, cond, 5e-5)
					wantTarget := make([]float32, 2)
					for i := range wantTarget {
						wantTarget[i] = time*inputGrad[i] - gv[i]
					}
					assertSliceClose(t, "CPU target", got.DTarget, wantTarget, 5e-5)
					if d.downloadCalls != 2 || d.downloadedElements != 10 || d.accumulations != len(legacy.ResidentBindings) {
						t.Fatal("scalar/result only downloads or accumulationcount", d.downloadCalls, d.downloadedElements, d.accumulations)
					}
					if sources.Input.Owner.Close() != nil || root.Close() != nil {
						t.Fatal("reverse close")
					}
					if _, e := state.Update(ctx, 100); e != nil {
						t.Fatal(e)
					}
					names := make([]string, 0, len(legacy.ResidentBindings))
					for name := range legacy.ResidentBindings {
						names = append(names, name)
					}
					sort.Strings(names)
					for i, name := range names {
						assertSliceClose(t, name, d.validatedGradients[i], flowHeadGradientMap(grad)[strings.TrimPrefix(name, "flow_lm.flow_net.")], 5e-5)
					}
					if state.Close() != nil {
						t.Fatal("close")
					}
				})
			}
		}
	}
}
func TestTrainingDiagonalDeviceInputsFailureOrdering(t *testing.T) {
	for _, kind := range []string{"row", "model", "root", "prediction", "condition", "extent", "logvar", "time", "scale", "budget", "flag", "witness", "alias", "cancel", "condition-nan", "input-inf", "prediction-inf", "desired-nan", "loss-overflow", "backward-fail"} {
		t.Run(kind, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			m, state, d, root, in := diagonalDeviceFixture(t, .3)
			sources, err := beginDiagonalDeviceSources(ctx, root, m, 2, diagonalDeviceRoot{root, 0}, in, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			time, logvar, scale, budget := in.Time, float32(0), float32(1), int64(64<<20)
			conditionSize := 3
			switch kind {
			case "row":
				sources.Row = 1
			case "model":
				m = tinyTrainableFlowHead()
			case "root":
				sources.Root = sources.Input.Owner
			case "prediction":
				sources.Prediction.Output++
			case "condition":
				sources.Condition.Owner = sources.Input.Owner
			case "extent":
				conditionSize++
			case "logvar":
				logvar = 81
			case "time":
				time = -1
			case "scale":
				scale = 2
			case "budget":
				budget = 16
			case "cancel":
				cancel()
			case "alias":
				d.afterDownload = func(v [][]float32) {
					if len(v) == 2 {
						v[1] = v[0]
					}
				}
			case "backward-fail":
				d.failTape = d.tapes + 2
			default:
				d.handoffAfterRun = func(p *hostResidentGraphTape) {
					if d.tapes != 3 {
						return
					}
					switch kind {
					case "flag":
						p.values[p.tape.Outputs[0]][0] = 1
					case "witness":
						p.values[p.tape.Outputs[1]][0] = float32(math.NaN())
					case "alias":
						p.values[p.tape.Outputs[1]] = p.values[p.tape.Outputs[0]]
					}
				}
			}
			// Mutate actual device root buffers, never fabricated host placeholders.
			if kind == "input-inf" || kind == "prediction-inf" || kind == "desired-nan" || kind == "loss-overflow" || kind == "condition-nan" {
				// The mock supplies retained arrays; capture them through read-only metadata
				// in the next admission construction and inject runtime invalid contents.
				d.handoffAfterRun = nil
				d.beforeRun = func(p *hostResidentGraphTape) {
					if d.tapes != 3 {
						return
					}
					id := 0
					value := float32(math.NaN())
					if kind == "input-inf" {
						id = 1
						value = float32(math.Inf(1))
					}
					if kind == "desired-nan" {
						id = 2
					}
					if kind == "prediction-inf" {
						id = 3
						value = float32(math.Inf(1))
					}
					if kind == "loss-overflow" {
						id = 3
						value = 1e30
					}
					p.borrowed[id][0] = value
				}
			}
			got, err := runDiagonalDeviceTraining(ctx, m, sources, 2, conditionSize, 2, time, logvar, scale, budget)
			if err == nil || !reflect.DeepEqual(got, DiagonalTrainingResult{}) || d.accumulations != 0 {
				t.Fatal("failure accumulated/published", err, d.accumulations)
			}
			before := d.tapes
			if _, e := runDiagonalDeviceTraining(ctx, m, sources, 2, 3, 2, in.Time, 0, 1, 64<<20); e == nil || d.tapes != before {
				t.Fatal("terminal retry")
			}
			if _, e := state.Update(ctx, 100); e == nil {
				t.Fatal("terminal update")
			}
			if _, e := state.Snapshot(ctx); e == nil {
				t.Fatal("terminal snapshot")
			}
			if root.Close() == nil || state.Close() == nil {
				t.Fatal("terminal ownership released")
			}
		})
	}
}

func TestTrainingDiagonalDeviceInputsSingleUse(t *testing.T) {
	ctx := context.Background()
	m, state, d, root, in := diagonalDeviceFixture(t, .3)
	sources, err := beginDiagonalDeviceSources(ctx, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	copySources := sources
	if _, err := runDiagonalDeviceTraining(ctx, m, sources, 0, 3, 2, .3, 0, .25, 64<<20); err != nil {
		t.Fatal(err)
	}
	before, count := d.tapes, d.accumulations
	if _, err := runDiagonalDeviceTraining(ctx, m, copySources, 0, 3, 2, .3, 0, .25, 64<<20); err == nil || d.tapes != before || d.accumulations != count {
		t.Fatal("duplicate accumulated")
	}
	if _, err := state.Update(ctx, 100); err == nil || root.Close() == nil {
		t.Fatal("duplicate did not terminally fence")
	}
}
func TestTrainingDiagonalDeviceInputsBuilderAdmission(t *testing.T) {
	for _, kind := range []string{"nil", "times", "condition", "channels", "output", "scale", "time", "logvar", "budget", "latebudget", "nonfiniteparam"} {
		t.Run(kind, func(t *testing.T) {
			m := tinyTrainableFlowHead()
			condition, n := 3, 2
			time, lv, scale, budget := float32(.3), float32(0), float32(1), int64(64<<20)
			switch kind {
			case "nil":
				m = nil
			case "times":
				m.Time = m.Time[:1]
			case "condition":
				condition++
			case "channels":
				n++
			case "output":
				m.Final.Linear.Out = 3
			case "scale":
				scale = float32(math.NaN())
			case "time":
				time = 2
			case "logvar":
				lv = -81
			case "budget":
				budget = 0
			case "latebudget":
				budget = 100
			case "nonfiniteparam":
				m.Input.Weight[0] = float32(math.Inf(1))
			}
			if p, err := m.buildDiagonalDeviceTrainingTape(condition, n, time, lv, scale, budget); err == nil || len(p.tape.Tensors) != 0 {
				t.Fatal("invalid builder admitted")
			}
		})
	}
	for _, kind := range []string{"cutoff", "extent", "budget"} {
		n, lv, budget := 2, float32(0), int64(64<<20)
		switch kind {
		case "cutoff":
			lv = 81
		case "extent":
			n = 0
		case "budget":
			budget = 16
		}
		if _, _, err := buildDiagonalDeviceAdmission(3, n, lv, budget); err == nil {
			t.Fatal("scalar graph admission", kind)
		}
	}
	m := tinyTrainableFlowHead()
	p, err := m.buildDiagonalDeviceTrainingTape(3, 2, .3, 0, .25, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err := m.buildDiagonalDeviceTrainingTape(3, 2, .3, 0, .25, bytes-1); err == nil {
		t.Fatal("fullsource logicalbudget excluded")
	}
	if _, err := m.buildDiagonalDeviceTrainingTape(3, 2, .3, 0, .25, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"duplicate", "range", "shape", "extra", "notexternal"} {
		q := p
		q.tape.Tensors = append([]train.TensorSpec(nil), p.tape.Tensors...)
		switch kind {
		case "duplicate":
			q.ids.Desired = q.ids.Prediction
		case "range":
			q.ids.Input = -1
		case "shape":
			q.conditionSize++
		case "extra":
			q.tape.Tensors = append(q.tape.Tensors, train.TensorSpec{Shape: []int{1}, External: true})
		case "notexternal":
			q.tape.Tensors[q.ids.Desired].External = false
		}
		if validateDiagonalDeviceIDs(q.tape, q.ids, q.conditionSize, q.channels, 64<<20) == nil {
			t.Fatal("source declaration", kind)
		}
	}
	t.Logf("backward fullsource bytes=%d commands=%d tensors=%d", bytes, len(p.tape.Commands), len(p.tape.Tensors))
}
func TestTrainingDiagonalDeviceInputsOwnerAdmission(t *testing.T) {
	if _, err := beginDiagonalDeviceSources(context.Background(), nil, nil, 0, diagonalDeviceRoot{}, DiagonalForwardInputs{}, 1024); err == nil {
		t.Fatal("nil parent")
	}
	if _, err := runDiagonalDeviceTraining(context.Background(), nil, diagonalDeviceSources{}, 0, 3, 2, .3, 0, 1, 1024); err == nil {
		t.Fatal("nil root")
	}
	for _, kind := range []string{"row", "owner", "forward", "foreign", "extent", "closed", "latebudget"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			m, state, d, root, in := diagonalDeviceFixture(t, .3)
			condition := diagonalDeviceRoot{root, 0}
			row := 0
			budget := int64(64 << 20)
			switch kind {
			case "row":
				row = -1
			case "owner":
				condition.Owner = nil
			case "forward":
				in.Target = nil
			case "foreign":
				_, _, _, condition.Owner, _ = diagonalDeviceFixture(t, .3)
			case "extent":
				condition.Output = 99
			case "closed":
				if root.Close() != nil {
					t.Fatal("fixtureclose")
				}
			case "latebudget":
				budget = 16
			}
			before := d.tapes
			if _, err := beginDiagonalDeviceSources(ctx, root, m, row, condition, in, budget); err == nil || d.tapes != before {
				t.Fatal("owner admission allocated")
			}
			if _, err := state.Update(ctx, 100); err == nil {
				t.Fatal("failure fenced")
			}
		})
	}
}
func TestTrainingDiagonalDeviceInputsRuntimeFailures(t *testing.T) {
	for _, kind := range []string{"nilcontext", "no-provenance", "resultcount", "resultshape", "resultnan", "resultalias", "negativesquare", "cancel-afterflag", "closechild"} {
		t.Run(kind, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			m, state, d, root, in := diagonalDeviceFixture(t, .3)
			sources, err := beginDiagonalDeviceSources(ctx, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			switch kind {
			case "nilcontext":
				ctx = nil
			case "no-provenance":
				sources.provenance = nil
			case "cancel-afterflag":
				d.afterDownload = func(v [][]float32) {
					if len(v) == 2 {
						cancel()
					}
				}
			case "closechild":
				d.handoffBeforeClose = func(p *hostResidentGraphTape) error {
					if len(p.tape.Outputs) == 3 && len(p.tape.Tensors[p.tape.Outputs[0]].Shape) == 1 && p.tape.Tensors[p.tape.Outputs[0]].Shape[0] == 2 {
						return fmt.Errorf("injected child close")
					}
					return nil
				}
			default:
				d.afterDownload = func(v [][]float32) {
					if len(v) != 3 {
						return
					}
					switch kind {
					case "resultcount":
						v[2] = nil
					case "resultshape":
						v[0] = nil
					case "resultnan":
						v[1][0] = float32(math.NaN())
					case "resultalias":
						v[1] = v[0][:2]
					case "negativesquare":
						v[2][0] = -1
					}
				}
			}
			got, err := runDiagonalDeviceTraining(ctx, m, sources, 0, 3, 2, .3, 0, 1, 64<<20)
			if err == nil || !reflect.DeepEqual(got, DiagonalTrainingResult{}) {
				t.Fatal("runtime failure published", err)
			}
			if _, e := state.Snapshot(context.Background()); e == nil {
				t.Fatal("publicationfence")
			}
			if _, e := state.Update(context.Background(), 100); e == nil || state.Close() == nil {
				t.Fatal("terminal ownership")
			}
		})
	}
}

func TestTrainingDiagonalDeviceInputsNumericalReference(t *testing.T) {
	var comparisons, sum, squares, maxError float64
	for _, time := range []float32{0, .3, 1} {
		for _, scale := range []float32{0, .25, 1} {
			for _, lv := range []float32{-2, 0, 1} {
				ctx := context.Background()
				m, state, d, root, in := diagonalDeviceFixture(t, time)
				sources, err := beginDiagonalDeviceSources(ctx, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				got, err := runDiagonalDeviceTraining(ctx, m, sources, 0, 3, 2, time, lv, scale, 64<<20)
				if err != nil {
					t.Fatal(err)
				}
				x, desired := make([]float32, 2), make([]float32, 2)
				for i := range x {
					x[i] = time*in.Target[i] + (1-time)*in.Noise[i]
					desired[i] = in.Target[i] - in.Noise[i]
				}
				prediction, _, _, _, _, err := m.ForwardBackward(in.Condition, []float32{time, time}, x, []float32{0, 0})
				if err != nil {
					t.Fatal(err)
				}
				var raw float64
				seed := make([]float32, 2)
				factor := float32(math.Exp(float64(lv))) / 2
				for i, p := range prediction {
					diff := p - desired[i]
					raw += float64(diff) * float64(diff)
					seed[i] = (2 * diff) * factor
					seed[i] *= scale
				}
				_, grad, cond, _, dx, err := m.ForwardBackward(in.Condition, []float32{time, time}, x, seed)
				if err != nil {
					t.Fatal(err)
				}
				compare := func(v, w []float32, gate float64) {
					t.Helper()
					if len(v) != len(w) {
						t.Fatal("refextent")
					}
					for i, x := range v {
						e := math.Abs(float64(x) - float64(w[i]))
						if e > gate {
							t.Fatal("CPU reference error", e)
						}
						comparisons++
						sum += e
						squares += e * e
						maxError = math.Max(maxError, e)
					}
				}
				compare(got.DCondition, cond, 5e-5)
				target := make([]float32, 2)
				for i := range target {
					target[i] = time*dx[i] - seed[i]
				}
				compare(got.DTarget, target, 5e-5)
				expectedStats := []float64{raw, raw*math.Exp(float64(lv))/2 - float64(lv), raw*math.Exp(float64(lv))/2 - 1}
				for i, v := range []float64{got.RawSquare, got.Loss, float64(got.DLogVariance)} {
					e := math.Abs(v - expectedStats[i])
					if e > 2e-6+2e-6*math.Abs(expectedStats[i]) {
						t.Fatal("F64loss", e)
					}
					comparisons++
					sum += e
					squares += e * e
					maxError = math.Max(maxError, e)
				}
				if root.Close() != nil {
					t.Fatal("close")
				}
				if _, err := state.Update(ctx, 100); err != nil {
					t.Fatal(err)
				}
				params := flowHeadGradientMap(grad)
				names := make([]string, 0, len(params))
				for name := range params {
					names = append(names, name)
				}
				sort.Strings(names)
				for i, name := range names {
					compare(d.validatedGradients[i], params[name], 5e-5)
				}
				if state.Close() != nil {
					t.Fatal("stateclose")
				}
			}
		}
	}
	t.Logf("Independent CPU VJP/F64 loss comparisons %.0f mean_abs %.9g RMS %.9g max_abs %.9g unchanged gates5e-5 gradients;2e-6+2e-6*abs loss", comparisons, sum/comparisons, math.Sqrt(squares/comparisons), maxError)
}
func TestTrainingDiagonalDeviceInputsAdmissionLargeWitness(t *testing.T) {
	tape, ids, err := buildDiagonalDeviceAdmission(16385, 2, 0, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	condition := make([]float32, 16385)
	condition[16384] = float32(math.NaN())
	output := interpretFlowTapeStorageBound(t, tape, true, map[int][]float32{ids.Condition: condition, ids.Input: []float32{0, 0}, ids.Desired: []float32{.2, .1}, ids.Prediction: []float32{.2, .1}})
	if output[0][0] != 0 || !math.IsNaN(float64(output[1][0])) {
		t.Fatal("lastchunk nonfinite hidden")
	}
	for i := range condition {
		condition[i] = math.MaxFloat32
	}
	output = interpretFlowTapeStorageBound(t, tape, true, map[int][]float32{ids.Condition: condition, ids.Input: []float32{0, 0}, ids.Desired: []float32{.2, .1}, ids.Prediction: []float32{.2, .1}})
	if output[1][0] != 0 {
		t.Fatal("finite max values overflowed witness")
	}
}

func TestTrainingDiagonalDeviceInputsFrozenMetadata(t *testing.T) {
	ctx := context.Background()
	m, state, d, root, in := diagonalDeviceFixture(t, .3)
	expected, err := m.BuildDiagonalForwardTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	f := interpretFlowTape(t, expected.Tape)
	backward, err := m.BuildDiagonalTrainingTape(in.Condition, .3, f[0], f[2], f[1], 0, .25, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	want := interpretFlowTape(t, backward.Tape)
	sources, err := beginDiagonalDeviceSources(ctx, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	m.Time[0].Frequencies[0] = 123
	m.Time[0].RMSEpsilon = .5
	m.Blocks[0].Epsilon = .3
	m.Final.Epsilon = .7
	m.Input.In = 999
	got, err := runDiagonalDeviceTraining(ctx, m, sources, 0, 3, 2, .3, 0, .25, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "frozen condition", got.DCondition, want[0], 2e-5)
	assertSliceClose(t, "frozen target", got.DTarget, want[1], 2e-5)
	if d.downloadCalls != 2 || root.Close() != nil || state.Close() != nil {
		t.Fatal("frozen graph ownership")
	}
}

func TestTrainingDiagonalDeviceInputsBoundaryGuards(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := DiagonalForwardInputs{Condition: []float32{.2, -.3, .5}, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, Time: .3}
	p, err := m.BuildDiagonalForwardTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"count", "duplicate", "extent", "invalidgraph"} {
		q := p
		q.Tape.Outputs = append([]int(nil), p.Tape.Outputs...)
		q.Tape.Tensors = append([]train.TensorSpec(nil), p.Tape.Tensors...)
		switch kind {
		case "count":
			q.Tape.Outputs = q.Tape.Outputs[:2]
		case "duplicate":
			q.Tape.Outputs[1] = q.Tape.Outputs[0]
		case "extent":
			if validateDiagonalDeviceForwardRoots(q, 3, 64<<20) == nil {
				t.Fatal("extent")
			}
			continue
		case "invalidgraph":
			q.Tape.Outputs[1] = -1
		}
		if validateDiagonalDeviceForwardRoots(q, 2, 64<<20) == nil {
			t.Fatal("forward roots guard", kind)
		}
	}
	_, _, _, root, _ := diagonalDeviceFixture(t, .3)
	if _, err := beginDiagonalDeviceSources(nil, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20); err == nil {
		t.Fatal("nil begincontext")
	}
	if freezeDiagonalDeviceFlow(nil) != nil {
		t.Fatal("nil frozenmodel")
	}
}
