package pockettts

import (
	"encoding/json"
	"fmt"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
)

func deviceFlowFixture(t *testing.T, index int) (*FlowHeadCPU, FlowDualTrainingTape, flowDeviceTrainingTape, map[int][]float32) {
	t.Helper()
	m := tinyTrainableFlowHead()
	in := primaryFixture()
	times := []float32{in.S, in.T}
	host, err := m.buildFlowDualTrainingTape(in.Condition, times, in.Input, index, in.GV, in.GT, 64<<20, true, true)
	if err != nil {
		t.Fatal(err)
	}
	device, err := m.buildFlowDeviceInputsTrainingTape(flowDeviceInputSizes{len(in.Condition), len(in.Input), len(in.GV)}, times, index, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	bound := map[int][]float32{device.inputs.Condition: in.Condition, device.inputs.Input: in.Input, device.inputs.ValueSeed: in.GV, device.inputs.TangentSeed: in.GT}
	return m, host, device, bound
}
func TestTrainingFlowDeviceInputsParity(t *testing.T) {
	for _, index := range []int{-1, 0, 1} {
		t.Run(string(rune('a'+index+1)), func(t *testing.T) {
			_, host, device, bound := deviceFlowFixture(t, index)
			if err := device.validate(64 << 20); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(host.Tape.Commands, device.flow.Tape.Commands) || !reflect.DeepEqual(host.Tape.Outputs, device.flow.Tape.Outputs) || !reflect.DeepEqual(host.ResidentBindings, device.flow.ResidentBindings) || len(host.Tape.Tensors) != len(device.flow.Tape.Tensors) {
				t.Fatal("graph command/ID/order/root drift")
			}
			for id, s := range device.flow.Tape.Tensors {
				if !reflect.DeepEqual(s.Shape, host.Tape.Tensors[id].Shape) {
					t.Fatal("shape drift")
				}
				if s.External {
					if s.Values != nil || bound[id] == nil {
						t.Fatal("shape-only declaration")
					}
				} else if !reflect.DeepEqual(s.Values, host.Tape.Tensors[id].Values) {
					t.Fatal("constant drift")
				}
			}
			before := map[int][]float32{}
			for id, v := range bound {
				before[id] = append([]float32(nil), v...)
			}
			hvalues := interpretFlowTape(t, host.Tape)
			dvalues := interpretFlowTapeStorageBound(t, device.flow.Tape, true, bound)
			for i := range dvalues {
				assertSliceClose(t, "device input output/adjoint", dvalues[i], hvalues[i], 2e-6)
			}
			if !reflect.DeepEqual(bound, before) {
				t.Fatal("borrowed inputs mutated")
			}
			result, err := device.flow.Decode(dvalues)
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Parameters) != len(device.flow.ParameterNames) || len(result.DTimes) != 2 {
				t.Fatal("gradient coverage")
			}
			// Complete source/inventory admission then stable pruning of all requested
			// roots and every parameter gradient. External IDs survive remapping.
			if err := train.ValidateResidentTapeBindings(device.flow.Tape, device.flow.ResidentBindings, 64<<20); err != nil {
				t.Fatal(err)
			}
			roots := append([]int(nil), device.flow.Tape.Outputs...)
			compact, mapping, err := train.PruneTape(device.flow.Tape, roots, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			remapped := map[int][]float32{}
			for id, v := range bound {
				if mapping[id] < 0 {
					t.Fatal("external pruned unexpectedly")
				}
				remapped[mapping[id]] = v
			}
			got := interpretFlowTapeStorageBound(t, compact, true, remapped)
			for i := range got {
				assertSliceClose(t, "pruned device input result", got[i], dvalues[i], 2e-6)
			}
		})
	}
}
func TestTrainingFlowDeviceInputsAdmission(t *testing.T) {
	for _, kind := range []string{"nil", "condition", "input", "output", "times", "selection", "negative-selection", "time-nan", "frequency", "angle", "weight-nan", "bias-nan", "bf16", "topology", "budget", "ceiling"} {
		t.Run(kind, func(t *testing.T) {
			m := tinyTrainableFlowHead()
			in := primaryFixture()
			sizes := flowDeviceInputSizes{3, 2, 2}
			times := []float32{in.S, in.T}
			index := 1
			budget := int64(64 << 20)
			switch kind {
			case "nil":
				m = nil
			case "condition":
				sizes.Condition++
			case "input":
				sizes.Input = 0
			case "output":
				sizes.Output++
			case "times":
				times = times[:1]
			case "selection":
				index = 2
			case "negative-selection":
				index = -2
			case "time-nan":
				times[0] = float32(math.NaN())
			case "frequency":
				m.Time[0].Frequencies[0] = float32(math.Inf(1))
			case "angle":
				times[0] = 1e8
			case "weight-nan":
				m.Input.Weight[0] = float32(math.NaN())
			case "bias-nan":
				m.Final.Linear.Bias[0] = float32(math.NaN())
			case "bf16":
				m.Input.WeightBF16 = []uint16{1}
			case "topology":
				m.Blocks[0].NormWeight = nil
			case "budget":
				budget = 0
			case "ceiling":
				budget = 16
			}
			got, err := m.buildFlowDeviceInputsTrainingTape(sizes, times, index, budget)
			if err == nil || len(got.flow.Tape.Tensors) != 0 {
				t.Fatal("invalid device builder admitted")
			}
		})
	}
	_, _, device, _ := deviceFlowFixture(t, 1)
	for _, kind := range []string{"duplicate", "range", "nonexternal", "extent", "inputprovenance", "extraexternal", "values"} {
		t.Run("descriptor-"+kind, func(t *testing.T) {
			p := device
			p.flow.Tape.Tensors = append([]train.TensorSpec(nil), device.flow.Tape.Tensors...)
			switch kind {
			case "duplicate":
				p.inputs.ValueSeed = p.inputs.Input
			case "range":
				p.inputs.Condition = -1
			case "nonexternal":
				p.flow.Tape.Tensors[p.inputs.Input].External = false
			case "extent":
				p.flow.conditionSize++
			case "inputprovenance":
				p.flow.inputValue++
			case "extraexternal":
				p.flow.Tape.Tensors = append(p.flow.Tape.Tensors, train.TensorSpec{Shape: []int{1}, External: true})
			case "values":
				p.flow.Tape.Tensors[p.inputs.Input].Values = []float32{0, 0}
			}
			if p.validate(64<<20) == nil {
				t.Fatal("tampered source declaration")
			}
		})
	}
}
func TestTrainingFlowDeviceInputsOwnership(t *testing.T) {
	m, _, device, _ := deviceFlowFixture(t, 1)
	param := device.flow.ResidentBindings["flow_lm.flow_net.input_proj.weight"].Parameter
	if &device.flow.Tape.Tensors[param].Values[0] != &m.Input.Weight[0] {
		t.Fatal("private parameter borrow lost")
	}
	times := []float32{.25, .8}
	before := append([]float32(nil), times...)
	a, err := m.buildFlowDeviceInputsTrainingTape(flowDeviceInputSizes{3, 2, 2}, times, 1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.buildFlowDeviceInputsTrainingTape(flowDeviceInputSizes{3, 2, 2}, times, 1, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	a.flow.Tape.Tensors[a.inputs.Input].Shape[0] = 99
	a.flow.Tape.Commands[0].Inputs[0] = 999
	if b.flow.Tape.Tensors[b.inputs.Input].Shape[0] != 2 || b.flow.Tape.Commands[0].Inputs[0] == 999 || !reflect.DeepEqual(times, before) {
		t.Fatal("graph/caller ownership")
	}
	// Public host builders still copy parameters and admit numeric values.
	in := primaryFixture()
	public, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, times, in.Input, 1, in.GV, in.GT, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	id := public.ResidentBindings["flow_lm.flow_net.input_proj.weight"].Parameter
	if &public.Tape.Tensors[id].Values[0] == &m.Input.Weight[0] {
		t.Fatal("public borrowed parameters")
	}
	for _, s := range public.Tape.Tensors {
		if s.External {
			t.Fatal("public shape-only route")
		}
	}
	in.Input[0] = float32(math.NaN())
	if _, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, times, in.Input, 1, in.GV, in.GT, 64<<20); err == nil {
		t.Fatal("host guard bypass")
	}
}
func TestTrainingFlowDeviceInputsPyTorchParity(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowHeadBackwardOracle
	if err := json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit {
		t.Fatal("oracle provenance")
	}
	for _, c := range oracle.Mixed {
		t.Run(fmt.Sprint(c.TimeIndex), func(t *testing.T) {
			m := flowHeadFromOracle(t, oracle.Parameters)
			device, err := m.buildFlowDeviceInputsTrainingTape(flowDeviceInputSizes{len(oracle.Condition), len(oracle.Input), len(c.DOutput)}, oracle.Times, c.TimeIndex, 64<<20)
			if err != nil {
				t.Fatal(err)
			}
			values := interpretFlowTapeStorageBound(t, device.flow.Tape, true, map[int][]float32{device.inputs.Condition: oracle.Condition, device.inputs.Input: oracle.Input, device.inputs.ValueSeed: c.DOutput, device.inputs.TangentSeed: c.DTangent})
			got, err := device.flow.Decode(values)
			if err != nil {
				t.Fatal(err)
			}
			assertSliceClose(t, "output", got.Output, oracle.Output, 3e-6)
			assertSliceClose(t, "JVP", got.Tangent, oracle.JVPTimes[c.TimeIndex], 5e-6)
			assertSliceClose(t, "condition", got.DCondition, c.DCondition, 8e-6)
			assertSliceClose(t, "input", got.DInput, c.DInput, 8e-6)
			assertSliceClose(t, "times", got.DTimes, c.DTimes, 8e-6)
			if len(got.Parameters) != len(c.Gradients) {
				t.Fatal("parameter coverage")
			}
			for name, gotParam := range got.Parameters {
				assertSliceClose(t, name, gotParam, c.Gradients[strings.TrimPrefix(name, "flow_lm.flow_net.")], 1e-5)
			}
		})
	}
}

func TestTrainingFlowDeviceInputsNumericalReference(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowHeadBackwardOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit || len(oracle.Mixed) != len(oracle.Times) {
		t.Fatal("oracle provenance/count")
	}
	var n, sum, squared, maximum float64
	for _, c := range oracle.Mixed {
		m := flowHeadFromOracle(t, oracle.Parameters)
		device, err := m.buildFlowDeviceInputsTrainingTape(flowDeviceInputSizes{len(oracle.Condition), len(oracle.Input), len(c.DOutput)}, oracle.Times, c.TimeIndex, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		values := interpretFlowTapeStorageBound(t, device.flow.Tape, true, map[int][]float32{device.inputs.Condition: oracle.Condition, device.inputs.Input: oracle.Input, device.inputs.ValueSeed: c.DOutput, device.inputs.TangentSeed: c.DTangent})
		got, err := device.flow.Decode(values)
		if err != nil {
			t.Fatal(err)
		}
		compare := func(got, want []float32, gate float64) {
			t.Helper()
			if len(got) != len(want) {
				t.Fatal("reference extent")
			}
			for i, x := range got {
				e := math.Abs(float64(x) - float64(want[i]))
				if e > gate {
					t.Fatal("reference error", e, gate)
				}
				n++
				sum += e
				squared += e * e
				maximum = math.Max(maximum, e)
			}
		}
		compare(got.Output, oracle.Output, 3e-6)
		compare(got.Tangent, oracle.JVPTimes[c.TimeIndex], 5e-6)
		compare(got.DCondition, c.DCondition, 8e-6)
		compare(got.DInput, c.DInput, 8e-6)
		compare(got.DTimes, c.DTimes, 8e-6)
		for name, v := range got.Parameters {
			compare(v, c.Gradients[strings.TrimPrefix(name, "flow_lm.flow_net.")], 1e-5)
		}
	}
	if n < 1 {
		t.Fatal("empty reference")
	}
	t.Logf("Independent pinned PyTorch comparisons %.0f mean_abs %.9g RMS %.9g max_abs %.9g unchanged per-result absolute gates3e-6/5e-6/8e-6/1e-5", n, sum/n, math.Sqrt(squared/n), maximum)
}

func TestTrainingFlowDeviceInputsInternalBranches(t *testing.T) {
	m := tinyTrainableFlowHead()
	times := []float32{.25, .8}
	s := flowDeviceTapeSources{sizes: flowDeviceInputSizes{3, 2, 2}}
	for _, kind := range []string{"host-condition", "host-input", "host-seed", "host-tangent", "host-time"} {
		t.Run(kind, func(t *testing.T) {
			var condition, input, gv, gt []float32
			var callback func(*flowTapeGraph, graphDual) (int, int)
			deviceTime := true
			switch kind {
			case "host-condition":
				condition = []float32{1}
			case "host-input":
				input = []float32{1}
			case "host-seed":
				gv = []float32{1}
			case "host-tangent":
				gt = []float32{1}
			case "host-time":
				deviceTime = false
			}
			if _, err := m.buildFlowDualTrainingTapeSources(condition, times, input, 1, gv, gt, 64<<20, deviceTime, true, callback, &s); err == nil {
				t.Fatal("ambiguous device/host contract")
			}
		})
	}
	in := primaryFixture()
	for _, kind := range []string{"seed-shape", "seed-nan", "selection"} {
		gt := append([]float32(nil), in.GT...)
		index := 1
		switch kind {
		case "seed-shape":
			gt = gt[:1]
		case "seed-nan":
			gt[0] = float32(math.NaN())
		case "selection":
			index = 3
		}
		if _, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, times, in.Input, index, in.GV, gt, 64<<20); err == nil {
			t.Fatal("legacy seed admission", kind)
		}
	}
}

func TestTrainingFlowDeviceInputsTopology(t *testing.T) {
	for _, kind := range []string{"final-topology", "condition-linear", "final-linear", "final-mod-linear", "time-first", "time-second", "time-topology", "block-first", "block-second", "block-mod-linear", "block-topology", "time-empty"} {
		t.Run(kind, func(t *testing.T) {
			m := tinyTrainableFlowHead()
			switch kind {
			case "final-topology":
				m.Final.Epsilon = 0
			case "condition-linear":
				m.Condition.Bias = []float32{0}
			case "final-linear":
				m.Final.Linear.Weight = nil
			case "final-mod-linear":
				m.Final.Modulation.WeightBF16 = []uint16{1}
			case "time-first":
				m.Time[0].FC1.Weight = nil
			case "time-second":
				m.Time[0].FC2.Weight = nil
			case "time-topology":
				m.Time[0].RMSEpsilon = 0
			case "block-first":
				m.Blocks[0].FC1.Weight = nil
			case "block-second":
				m.Blocks[0].FC2.Weight = nil
			case "block-mod-linear":
				m.Blocks[0].Modulation.Weight = nil
			case "block-topology":
				m.Blocks[0].NormBias = nil
			case "time-empty":
				m.Time[0].Frequencies = nil
				m.Time[0].FC1.In = 0
			}
			if _, err := m.buildFlowDeviceInputsTrainingTape(flowDeviceInputSizes{3, 2, 2}, []float32{.25, .8}, 1, 64<<20); err == nil {
				t.Fatal("topology admitted")
			}
		})
	}
}
func TestTrainingFlowDeviceInputsExactBudget(t *testing.T) {
	m, host, device, _ := deviceFlowFixture(t, 1)
	var hostBytes, deviceBytes, externalBytes int64
	for _, s := range host.Tape.Tensors {
		n, _ := s.Elements()
		hostBytes += int64(n) * 4
	}
	for _, s := range device.flow.Tape.Tensors {
		n, _ := s.Elements()
		deviceBytes += int64(n) * 4
		if s.External {
			externalBytes += int64(n) * 4
		}
	}
	if hostBytes != deviceBytes || externalBytes != 36 {
		t.Fatal("full logical external charge", hostBytes, deviceBytes, externalBytes)
	}
	times := []float32{.25, .8}
	if _, err := m.buildFlowDeviceInputsTrainingTape(flowDeviceInputSizes{3, 2, 2}, times, 1, deviceBytes-1); err == nil {
		t.Fatal("budget excluded device sources")
	}
	if _, err := m.buildFlowDeviceInputsTrainingTape(flowDeviceInputSizes{3, 2, 2}, times, 1, deviceBytes); err != nil {
		t.Fatal("exact logical ceiling rejected", err)
	}
	t.Logf("tiny fullsource bytes host=%d device=%d external=%d commands=%d tensors=%d unchanged native allocations/peak/latency unmeasured", hostBytes, deviceBytes, externalBytes, len(device.flow.Tape.Commands), len(device.flow.Tape.Tensors))
}

func TestTrainingFlowDeviceInputsTimeShape(t *testing.T) {
	m := tinyTrainableFlowHead()
	m.Time[0].Frequencies = make([]float32, 8193)
	m.Time[0].FC1.In = 16386
	m.Time[0].FC1.Weight = make([]float32, m.Time[0].FC1.In*m.Time[0].FC1.Out)
	if _, err := m.buildFlowDeviceInputsTrainingTape(flowDeviceInputSizes{3, 2, 2}, []float32{.25, .8}, 1, 64<<20); err == nil {
		t.Fatal("time kernel shape ceiling")
	}
}
