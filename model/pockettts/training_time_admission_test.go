package pockettts

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestTrainingFlowDeviceTimeAdmission(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowHeadBackwardOracle
	if err := json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	build := func(m *FlowHeadCPU, budget int64) (FlowDualTrainingTape, error) {
		return m.BuildFlowDeviceTimeTrainingTape(oracle.Condition, oracle.Times, oracle.Input, -1, oracle.DOutput, make([]float32, len(oracle.Output)), budget)
	}
	m := flowHeadFromOracle(t, oracle.Parameters)
	tape, err := build(m, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	result, err := tape.Decode(interpretFlowTape(t, tape.Tape))
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "unselected time tangent", result.Tangent, make([]float32, len(oracle.Output)), 0)
	cpuTape, err := m.BuildFlowDualTrainingTape(oracle.Condition, oracle.Times, oracle.Input, -1, oracle.DOutput, make([]float32, len(oracle.Output)), 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := cpuTape.Decode(interpretFlowTape(t, cpuTape.Tape))
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "unselected time scalar gradient", result.DTimes, cpu.DTimes, 8e-6)
	if _, err := build(m, 0); err == nil {
		t.Fatal("zero budget")
	}
	if _, err := build(m, 1); err == nil {
		t.Fatal("small budget")
	}
	for _, f := range []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.MaxFloat32), 1e20} {
		m := flowHeadFromOracle(t, oracle.Parameters)
		m.Time[0].Frequencies[0] = f
		if _, err := build(m, 64<<20); err == nil {
			t.Fatal("bad frequency")
		}
	}
	times := append([]float32(nil), oracle.Times...)
	times[0] = 1e7
	if _, err := m.BuildFlowDeviceTimeTrainingTape(oracle.Condition, times, oracle.Input, -1, oracle.DOutput, make([]float32, len(oracle.Output)), 64<<20); err == nil {
		t.Fatal("angle ceiling")
	}
	values := interpretFlowTape(t, tape.Tape)
	if _, err := tape.Decode(values[:4]); err == nil {
		t.Fatal("short decode")
	}
	if _, err := tape.decodeBoundary(values[:4]); err == nil {
		t.Fatal("short boundary")
	}
	values[4] = []float32{float32(math.NaN())}
	if _, err := tape.decodeBoundary(values[:tape.boundarySize()]); err == nil {
		t.Fatal("nonfinite time boundary")
	}
}
