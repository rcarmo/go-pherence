//go:build linux

package pockettts

import (
	"context"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"reflect"
	"sort"
	"testing"
)

func distillDeviceFixture(t *testing.T, s, tim float32) (*FlowHeadCPU, *train.ResidentState, *hostResidentGraphDriver, *train.ResidentTapeLease, DistillForwardInputs) {
	t.Helper()
	m := tinyTrainableFlowHead()
	in := DistillForwardInputs{Condition: []float32{.2, -.3, .5}, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, S: s, T: tim}
	graph, err := m.buildDistillDeviceTrainingTape(3, 2, s, tim, 0, in.Target, in.Noise, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	d := &hostResidentGraphDriver{t: t}
	snapshot := train.ResidentSnapshot{Config: train.ResidentConfig{LearningRate: .001, Beta1: .9, Beta2: .99, Epsilon: 1e-8, EMADecay: .9}, Parameters: map[string]train.ResidentParameter{}}
	for name, b := range graph.bindings {
		p := graph.tape.Tensors[b.Parameter].Values
		snapshot.Parameters[name] = train.ResidentParameter{Parameter: p, M: make([]float32, len(p)), V: make([]float32, len(p)), EMA: append([]float32(nil), p...)}
	}
	state, err := train.NewResidentState(context.Background(), d, snapshot, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{3}, Values: in.Condition}}, Outputs: []int{0}}
	root, err := state.BeginZeroPrunedTapeLease(context.Background(), tape, nil, tape.Outputs, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	return m, state, d, root, in
}
func distillHostBoundary(t *testing.T, m *FlowHeadCPU, in DistillForwardInputs, logvar float32) (DistillTrainingInputs, DistillTrainingTape, [][]float32) {
	t.Helper()
	f, err := m.BuildDistillForwardTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	v := interpretFlowTape(t, f.Tape)
	inputs := DistillTrainingInputs{Condition: in.Condition, PrimaryInput: v[0], Velocity: v[1], Tangent: v[2], EndpointInput: v[3], Desired: v[4], Prediction: v[5], Target: in.Target, Noise: in.Noise, S: in.S, T: in.T, LogVariance: logvar}
	p, err := m.BuildDistillTrainingTape(inputs, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	return inputs, p, interpretFlowTape(t, p.Tape)
}
func TestTrainingDistillDeviceInputsParity(t *testing.T) {
	for _, times := range [][2]float32{{0, 0}, {0, 1}, {.3, .8}, {1, 1}} {
		for _, lv := range []float32{-2, 0, 1} {
			for _, scale := range []float32{0, .25, 1} {
				t.Run(fmt.Sprint(times, lv, scale), func(t *testing.T) {
					ctx := context.Background()
					m, state, d, root, in := distillDeviceFixture(t, times[0], times[1])
					inputs, _, _ := distillHostBoundary(t, m, in, lv)
					legacy, err := m.buildScaledDistillTrainingTape(inputs, 64<<20, false, scale)
					if err != nil {
						t.Fatal(err)
					}
					want := interpretFlowTape(t, legacy.Tape)
					sources, err := beginDistillDeviceSources(ctx, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20)
					if err != nil {
						t.Fatal(err)
					}
					if d.downloadCalls != 0 {
						t.Fatal("forward root downloaded")
					}
					got, err := runDistillDeviceTraining(ctx, m, sources, 0, lv, scale, 64<<20)
					if err != nil {
						t.Fatal(err)
					}
					for i, v := range [][]float32{got.DCondition, got.DTarget, got.DNoise, got.DTimes, {float32(got.RawSquare), float32(got.Loss), got.DLogVariance}} {
						assertSliceClose(t, "distill result", v, want[i], 2e-5)
					}
					if d.downloadCalls != 2 || d.downloadedElements != 14 || d.accumulations != len(legacy.ResidentBindings) {
						t.Fatal("transfer/gradient accounting", d.downloadCalls, d.downloadedElements, d.accumulations)
					}
					if root.Close() != nil {
						t.Fatal("closeparent")
					}
					if _, err := state.Update(ctx, 100); err != nil {
						t.Fatal(err)
					}
					names := make([]string, 0, len(legacy.ResidentBindings))
					for name := range legacy.ResidentBindings {
						names = append(names, name)
					}
					sort.Strings(names)
					for i, name := range names {
						index := 0
						for _, id := range legacy.Tape.Outputs {
							if id == legacy.ResidentBindings[name].Gradient {
								break
							}
							index++
						}
						expected := append([]float32(nil), want[index]...)

						assertSliceClose(t, "primary-only "+name, d.validatedGradients[i], expected, 2e-5)
					}
					if state.Close() != nil {
						t.Fatal("closestate")
					}
				})
			}
		}
	}
}
func TestTrainingDistillDeviceInputsFailures(t *testing.T) {
	for _, kind := range []string{"row", "model", "root", "condition", "forward", "no-provenance", "nilcontext", "cancel", "logvar", "scale", "budget", "flag", "witness", "scalaralias", "input-inf", "velocity-nan", "tangent-inf", "endpoint-inf", "desired-nan", "prediction-overflow", "condition-nan", "backward-fail", "resultshape", "resultnan", "resultalias", "negative-square", "cancel-afterflag", "childclose"} {
		t.Run(kind, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			m, state, d, root, in := distillDeviceFixture(t, .3, .8)
			sources, err := beginDistillDeviceSources(ctx, root, m, 2, diagonalDeviceRoot{root, 0}, in, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			row, lv, scale, budget := 2, float32(0), float32(1), int64(64<<20)
			switch kind {
			case "row":
				row = 1
			case "model":
				m = tinyTrainableFlowHead()
			case "root":
				sources.root = sources.forward[0].Owner
			case "condition":
				sources.condition.Output++
			case "forward":
				sources.forward[0].Output++
			case "no-provenance":
				sources.provenance = nil
			case "nilcontext":
				ctx = nil
			case "cancel":
				cancel()
			case "logvar":
				lv = 81
			case "scale":
				scale = -1
			case "budget":
				budget = 16
			case "backward-fail":
				d.failTape = d.tapes + 2
			case "scalaralias":
				d.afterDownload = func(v [][]float32) {
					if len(v) == 2 {
						v[1] = v[0]
					}
				}
			case "cancel-afterflag":
				d.afterDownload = func(v [][]float32) {
					if len(v) == 2 {
						cancel()
					}
				}
			case "childclose":
				d.handoffBeforeClose = func(p *hostResidentGraphTape) error {
					if len(p.tape.Outputs) == 6 {
						return fmt.Errorf("injected child close")
					}
					return nil
				}
			case "resultshape", "resultnan", "resultalias", "negative-square":
				d.afterDownload = func(v [][]float32) {
					if len(v) != 5 {
						return
					}
					switch kind {
					case "resultshape":
						v[0] = nil
					case "resultnan":
						v[1][0] = float32(math.NaN())
					case "resultalias":
						v[2] = v[1]
					case "negative-square":
						v[4][0] = -1
					}
				}
			case "flag", "witness":
				d.afterDownload = func(v [][]float32) {
					if len(v) == 2 {
						if kind == "flag" {
							v[0][0] = 1
						} else {
							v[1][0] = float32(math.NaN())
						}
					}
				}
			default:
				d.beforeRun = func(p *hostResidentGraphTape) {
					if d.tapes != 3 {
						return
					}
					id := 0
					value := float32(math.NaN())
					switch kind {
					case "input-inf":
						id = 1
						value = float32(math.Inf(1))
					case "velocity-nan":
						id = 2
					case "tangent-inf":
						id = 3
						value = float32(math.Inf(1))
					case "endpoint-inf":
						id = 4
						value = float32(math.Inf(1))
					case "desired-nan":
						id = 5
					case "prediction-overflow":
						id = 6
						value = 1e30
					}
					p.borrowed[id][0] = value
				}
			}
			got, err := runDistillDeviceTraining(ctx, m, sources, row, lv, scale, budget)
			if err == nil || !reflect.DeepEqual(got, DistillTrainingResult{}) {
				t.Fatal("failure published", err)
			}
			late := kind == "resultshape" || kind == "resultnan" || kind == "resultalias" || kind == "negative-square" || kind == "childclose"
			if !late && d.accumulations != 0 {
				t.Fatal("admission failure accumulated")
			}
			before := d.tapes
			if _, e := runDistillDeviceTraining(context.Background(), m, sources, 2, 0, 1, 64<<20); e == nil || d.tapes != before {
				t.Fatal("terminal retry")
			}
			if _, e := state.Update(context.Background(), 100); e == nil {
				t.Fatal("terminalupdate")
			}
			if _, e := state.Snapshot(context.Background()); e == nil {
				t.Fatal("terminalsnapshot")
			}
			if root.Close() == nil || state.Close() == nil {
				t.Fatal("released uncertainowner")
			}
		})
	}
}

func TestTrainingDistillDeviceInputsSingleUseFrozen(t *testing.T) {
	ctx := context.Background()
	m, state, d, root, in := distillDeviceFixture(t, .3, .8)
	inputs, _, _ := distillHostBoundary(t, m, in, 0)
	scaled, e := m.buildScaledDistillTrainingTape(inputs, 64<<20, false, .25)
	if e != nil {
		t.Fatal(e)
	}
	expected := interpretFlowTape(t, scaled.Tape)
	sources, err := beginDistillDeviceSources(ctx, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	copySources := sources
	m.Time[0].Frequencies[0] = 123
	m.Time[0].RMSEpsilon = .5
	m.Blocks[0].Epsilon = .3
	m.Final.Epsilon = .7
	m.Input.In = 999
	in.Target[0] = 55
	in.Noise[0] = -66
	got, err := runDistillDeviceTraining(ctx, m, sources, 0, 0, .25, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range [][]float32{got.DCondition, got.DTarget, got.DNoise, got.DTimes} {
		assertSliceClose(t, "frozen metadata/targetnoise", v, expected[i], 2e-5)
	}
	before, count := d.tapes, d.accumulations
	if _, err := runDistillDeviceTraining(ctx, m, copySources, 0, 0, .25, 64<<20); err == nil || d.tapes != before || d.accumulations != count {
		t.Fatal("duplicate gradient")
	}
	if _, err := state.Update(ctx, 100); err == nil || root.Close() == nil {
		t.Fatal("duplicate fence")
	}
}
func TestTrainingDistillDeviceInputsBuilderAdmission(t *testing.T) {
	for _, kind := range []string{"nil", "timeorder", "time-nan", "condition", "channels", "times", "logvar", "budget", "latebudget", "target", "noise", "paramnan"} {
		t.Run(kind, func(t *testing.T) {
			m := tinyTrainableFlowHead()
			c, n, s, tim, lv, budget := 3, 2, float32(.3), float32(.8), float32(0), int64(64<<20)
			target, noise := []float32{.4, .1}, []float32{.1, -.2}
			switch kind {
			case "nil":
				m = nil
			case "timeorder":
				s = .9
			case "time-nan":
				tim = float32(math.NaN())
			case "condition":
				c++
			case "channels":
				n++
			case "times":
				m.Time = m.Time[:1]
			case "logvar":
				lv = 81
			case "budget":
				budget = 0
			case "latebudget":
				budget = 100
			case "target":
				target = nil
			case "noise":
				noise[0] = float32(math.NaN())
			case "paramnan":
				m.Input.Weight[0] = float32(math.Inf(1))
			}
			if p, err := m.buildDistillDeviceTrainingTape(c, n, s, tim, lv, target, noise, budget); err == nil || len(p.tape.Tensors) != 0 {
				t.Fatal("invalid builder")
			}
		})
	}
	m := tinyTrainableFlowHead()
	p, err := m.buildDistillDeviceTrainingTape(3, 2, .3, .8, 0, []float32{.4, .1}, []float32{.1, -.2}, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	var bytes int64
	for _, s := range p.tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, err := m.buildDistillDeviceTrainingTape(3, 2, .3, .8, 0, []float32{.4, .1}, []float32{.1, -.2}, bytes-1); err == nil {
		t.Fatal("combinedfullsourcebudget")
	}
	if _, err := m.buildDistillDeviceTrainingTape(3, 2, .3, .8, 0, []float32{.4, .1}, []float32{.1, -.2}, bytes); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"duplicate", "range", "shape", "extraexternal"} {
		q := p
		q.tape.Tensors = append([]train.TensorSpec(nil), p.tape.Tensors...)
		switch kind {
		case "duplicate":
			q.ids.Forward[0] = q.ids.Condition
		case "range":
			q.ids.Condition = -1
		case "shape":
			q.tape.Tensors[q.ids.Forward[0]].Shape = []int{1, 2}
		case "extraexternal":
			q.tape.Tensors = append(q.tape.Tensors, train.TensorSpec{Shape: []int{1}, External: true})
		}
		if validateDistillDeviceIDs(q.tape, q.ids, 3, 2, 64<<20) == nil {
			t.Fatal("source declaration", kind)
		}
	}
	t.Logf("tiny composed backward fullsource bytes=%d commands=%d tensors=%d externalroots=7 primaryparams=%d", bytes, len(p.tape.Commands), len(p.tape.Tensors), len(p.bindings))
}
func TestTrainingDistillDeviceInputsOwnerAdmission(t *testing.T) {
	if _, err := beginDistillDeviceSources(context.Background(), nil, nil, 0, diagonalDeviceRoot{}, DistillForwardInputs{}, 1024); err == nil {
		t.Fatal("nil root")
	}
	if _, err := runDistillDeviceTraining(context.Background(), nil, distillDeviceSources{}, 0, 0, 1, 1024); err == nil {
		t.Fatal("nilrunroot")
	}
	for _, kind := range []string{"nilcontext", "row", "owner", "forward", "foreign", "extent", "closed", "latebudget"} {
		t.Run(kind, func(t *testing.T) {
			ctx := context.Background()
			m, state, d, root, in := distillDeviceFixture(t, .3, .8)
			condition := diagonalDeviceRoot{root, 0}
			row := 0
			budget := int64(64 << 20)
			switch kind {
			case "nilcontext":
				ctx = nil
			case "row":
				row = -1
			case "owner":
				condition.Owner = nil
			case "forward":
				in.Target = nil
			case "foreign":
				_, _, _, condition.Owner, _ = distillDeviceFixture(t, .3, .8)
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
			if _, err := beginDistillDeviceSources(ctx, root, m, row, condition, in, budget); err == nil || d.tapes != before {
				t.Fatal("owner admission allocated")
			}
			if _, err := state.Update(context.Background(), 100); err == nil {
				t.Fatal("failure fence")
			}
		})
	}
}
func TestTrainingDistillDeviceInputsForwardBoundary(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := DistillForwardInputs{Condition: []float32{.2, -.3, .5}, Target: []float32{.4, .1}, Noise: []float32{.1, -.2}, S: .3, T: .8}
	p, err := m.BuildDistillForwardTape(in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"count", "duplicate", "extent", "invalidgraph"} {
		q := p
		q.Tape.Outputs = append([]int(nil), p.Tape.Outputs...)
		switch kind {
		case "count":
			q.Tape.Outputs = q.Tape.Outputs[:5]
		case "duplicate":
			q.Tape.Outputs[1] = q.Tape.Outputs[0]
		case "extent":
			if validateDistillDeviceForwardRoots(q, 3, 64<<20) == nil {
				t.Fatal("extent")
			}
			continue
		case "invalidgraph":
			q.Tape.Outputs[1] = -1
		}
		if validateDistillDeviceForwardRoots(q, 2, 64<<20) == nil {
			t.Fatal("forward rootcontract", kind)
		}
	}
}
func TestTrainingDistillDeviceInputsNumericalReference(t *testing.T) {
	var count, sum, squares, maximum float64
	for _, times := range [][2]float32{{0, 0}, {0, 1}, {.3, .8}, {1, 1}} {
		for _, lv := range []float32{-2, 0, 1} {
			ctx := context.Background()
			m, state, d, root, in := distillDeviceFixture(t, times[0], times[1])
			sources, err := beginDistillDeviceSources(ctx, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			got, err := runDistillDeviceTraining(ctx, m, sources, 0, lv, .25, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			// Independent CPU forward, endpoint input-only reverse, primary mixed AD.
			x := make([]float32, 2)
			for i := range x {
				x[i] = in.S*in.Target[i] + (1-in.S)*in.Noise[i]
			}
			velocity, tangent, err := m.ForwardTimeJVP(in.Condition, []float32{in.S, in.T}, x, 1)
			if err != nil {
				t.Fatal(err)
			}
			delta := in.T - in.S
			xt, desired := make([]float32, 2), make([]float32, 2)
			for i := range xt {
				xt[i] = x[i] + delta*velocity[i]
				desired[i] = velocity[i] + delta*tangent[i]
			}
			tape, pred, err := m.forwardTraining(in.Condition, []float32{in.T, in.T}, xt)
			if err != nil {
				t.Fatal(err)
			}
			dr, de := make([]float32, 2), make([]float32, 2)
			var square float64
			for i, p := range pred {
				diff := float64(desired[i]) - float64(p)
				square += diff * diff
				dr[i] = float32(2 * diff * math.Exp(float64(lv)) / 2)
				de[i] = -dr[i]
			}
			_, _, dx, err := m.backwardTraining(tape, de, newFlowHeadGradients(m))
			if err != nil {
				t.Fatal(err)
			}
			gv, gt := make([]float32, 2), make([]float32, 2)
			dd := float32(0)
			for i := range gv {
				gv[i] = (dr[i] + delta*dx[i]) * .25
				gt[i] = (delta * dr[i]) * .25
				dx[i] *= .25
				dd += dr[i]*tangent[i] + dx[i]*velocity[i]
			}
			_, _, grad, dc, dt, dp, err := m.ForwardTimeJVPBackward(in.Condition, []float32{in.S, in.T}, x, 1, gv, gt)
			if err != nil {
				t.Fatal(err)
			}
			target, noise := make([]float32, 2), make([]float32, 2)
			ds := float32(0)
			for i := range target {
				v := dp[i] + dx[i]
				target[i] = in.S * v
				noise[i] = (1 - in.S) * v
				ds += v * (in.Target[i] - in.Noise[i])
			}
			ds += dt[0] - dd
			compare := func(a, b []float32, gate float64) {
				t.Helper()
				if len(a) != len(b) {
					t.Fatal("refextent")
				}
				for i, x := range a {
					err := math.Abs(float64(x) - float64(b[i]))
					if err > gate {
						t.Fatal("CPU reference", err, gate)
					}
					count++
					sum += err
					squares += err * err
					maximum = math.Max(maximum, err)
				}
			}
			compare(got.DCondition, dc, 5e-5)
			compare(got.DTarget, target, 5e-5)
			compare(got.DNoise, noise, 5e-5)
			compare(got.DTimes, []float32{ds, dt[1] + dd}, 5e-5)
			expectedStats := []float64{square, square*math.Exp(float64(lv))/2 - float64(lv), square*math.Exp(float64(lv))/2 - 1}
			for i, v := range []float64{got.RawSquare, got.Loss, float64(got.DLogVariance)} {
				err := math.Abs(v - expectedStats[i])
				if err > 2e-6+2e-6*math.Abs(expectedStats[i]) {
					t.Fatal("F64 loss", err)
				}
				count++
				sum += err
				squares += err * err
				maximum = math.Max(maximum, err)
			}
			if root.Close() != nil {
				t.Fatal("closeparent")
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
				wanted := append([]float32(nil), params[name]...)

				compare(d.validatedGradients[i], wanted, 5e-5)
			}
			if state.Close() != nil {
				t.Fatal("stateclose")
			}
		}
	}
	t.Logf("Independent CPU mixedAD/F64 loss comparisons %.0f mean_abs %.9g RMS %.9g max_abs %.9g unchanged gates5e-5 gradients;2e-6+2e-6*abs loss", count, sum/count, math.Sqrt(squares/count), maximum)
}

func TestTrainingDistillDeviceInputsScalarGuards(t *testing.T) {
	for _, kind := range []string{"logvar", "extent", "condition", "budget"} {
		n, c, lv, budget := 2, 3, float32(0), int64(64<<20)
		switch kind {
		case "logvar":
			lv = 81
		case "extent":
			n = 0
		case "condition":
			c = 0
		case "budget":
			budget = 16
		}
		if _, _, err := buildDistillDeviceAdmission(c, n, lv, budget); err == nil {
			t.Fatal("scalar admission", kind)
		}
	}
	tape, ids, err := buildDistillDeviceAdmission(3, 2, 0, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	values := map[int][]float32{ids.Condition: {.2, -.3, .5}}
	for _, id := range ids.Forward {
		values[id] = []float32{.1, .2}
	}
	var bytes int64
	for _, s := range tape.Tensors {
		n, _ := s.Elements()
		bytes += int64(n) * 4
	}
	if _, _, err := buildDistillDeviceAdmission(3, 2, 0, bytes-1); err == nil {
		t.Fatal("exact scalar budget")
	}
	if _, _, err := buildDistillDeviceAdmission(3, 2, 0, bytes); err != nil {
		t.Fatal(err)
	}
	for i, id := range ids.Forward {
		for _, bad := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))} {
			values[id][0] = bad
			out := interpretFlowTapeStorageBound(t, tape, true, values)
			if out[0][0] == 0 && isFinite(out[1][0]) && out[1][0] == 0 {
				t.Fatal("unsafe scalaraccept", i)
			}
			values[id][0] = .1
		}
	}
	values[ids.Condition][0] = float32(math.Inf(1))
	out := interpretFlowTapeStorageBound(t, tape, true, values)
	if isFinite(out[1][0]) {
		t.Fatal("nonfinitecondition")
	}
	q := tape
	q.Tensors = append([]train.TensorSpec(nil), tape.Tensors...)
	q.Tensors[ids.Forward[0]].External = false
	if validateDistillDeviceIDs(q, ids, 3, 2, 64<<20) == nil {
		t.Fatal("nonexternal source")
	}
}
func TestTrainingDistillDeviceInputsLateRowRetention(t *testing.T) {
	ctx := context.Background()
	m, state, d, root, in := distillDeviceFixture(t, .3, .8)
	first, err := beginDistillDeviceSources(ctx, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = runDistillDeviceTraining(ctx, m, first, 0, 0, .25, 64<<20); err != nil {
		t.Fatal(err)
	}
	count := d.accumulations
	if count == 0 {
		t.Fatal("first row no gradients")
	}
	second, err := beginDistillDeviceSources(ctx, root, m, 1, diagonalDeviceRoot{root, 0}, in, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	d.afterDownload = func(v [][]float32) {
		if len(v) == 2 {
			v[0][0] = 1
		}
	}
	if r, e := runDistillDeviceTraining(ctx, m, second, 1, 0, .25, 64<<20); e == nil || !reflect.DeepEqual(r, DistillTrainingResult{}) || d.accumulations != count {
		t.Fatal("late row leaked update/gradients")
	}
	before := d.tapes
	if _, e := runDistillDeviceTraining(ctx, m, second, 1, 0, .25, 64<<20); e == nil || d.tapes != before {
		t.Fatal("late retry")
	}
	if _, e := state.Snapshot(ctx); e == nil {
		t.Fatal("late snapshot")
	}
	if _, e := state.Update(ctx, 100); e == nil {
		t.Fatal("late update")
	}
	if state.Close() == nil || root.Close() == nil {
		t.Fatal("late uncertainrelease")
	}
}
func TestTrainingDistillDeviceInputsBeginCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	m, _, d, root, in := distillDeviceFixture(t, .3, .8)
	cancel()
	before := d.tapes
	if _, err := beginDistillDeviceSources(ctx, root, m, 0, diagonalDeviceRoot{root, 0}, in, 64<<20); err == nil || d.tapes != before || root.Close() == nil {
		t.Fatal("begin cancelled construction/retention")
	}
}
