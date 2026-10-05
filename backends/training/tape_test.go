package training

import (
	"math"
	"testing"
)

func vectorTapeFixture() Tape {
	return Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{3, 4}}, {Shape: []int{2}}}, Commands: []Command{{Op: "vector", Inputs: []int{0, 1}, Output: 2, Scale: 1}}, Outputs: []int{2}}
}
func TestTrainingTapeAdmission(t *testing.T) {
	if err := vectorTapeFixture().Validate(24); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Tape){
		func(t *Tape) { t.Tensors[0].Values[0] = float32(math.NaN()) }, func(t *Tape) { t.Tensors[0].Values = nil }, func(t *Tape) { t.Commands[0].Output = 0 }, func(t *Tape) { t.Commands[0].Inputs[0] = 3 }, func(t *Tape) { t.Commands[0].Op = "unknown" }, func(t *Tape) { t.Commands = append(t.Commands, t.Commands[0]) }, func(t *Tape) { t.Outputs[0] = 3 }, func(t *Tape) { t.Tensors[2].Shape = []int{3} }, func(t *Tape) { t.Commands[0].Inputs = nil }, func(t *Tape) { t.Tensors[0].Values = []float32{1} },
	} {
		tape := vectorTapeFixture()
		mutate(&tape)
		if err := tape.Validate(1024); err == nil {
			t.Fatal("malformed graph admitted")
		}
	}
	if err := vectorTapeFixture().Validate(23); err == nil {
		t.Fatal("budget ignored")
	}
	tape := vectorTapeFixture()
	tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{1, 2}})
	tape.Commands = append(tape.Commands, Command{Op: "reshape", Inputs: []int{2}, Output: 3})
	tape.Outputs = []int{3}
	if err := tape.Validate(1024); err != nil {
		t.Fatal(err)
	}
	tape.Tensors[3].Shape = []int{1, 3}
	if err := tape.Validate(1024); err == nil {
		t.Fatal("reshape count admitted")
	}
}
func TestTrainingTapeVectorBounds(t *testing.T) {
	for _, s := range []struct {
		n, source, offset int
		mode              uint32
	}{{3, 3, 0, 0}, {2, 4, 1, 1}, {4, 2, 1, 2}} {
		if err := VectorShape(s.n, s.source, s.offset, s.mode, 1); err != nil {
			t.Fatal(err)
		}
	}
	for _, s := range []struct {
		n, source, offset int
		mode              uint32
	}{{0, 1, 0, 0}, {2, 3, 0, 0}, {2, 3, 2, 1}, {3, 2, 2, 2}, {2, 2, -1, 0}, {2, 2, 0, 3}} {
		if err := VectorShape(s.n, s.source, s.offset, s.mode, 1); err == nil {
			t.Fatal("vector bound")
		}
	}
}

// Isolate the complete finite-value admission scan identified in released
// profiles; the graph and values are prepared outside the benchmark region.
func BenchmarkTrainingTapeValidateConstants(b *testing.B) {
	const n = 1 << 20
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i%257-128) * .03125
	}
	tape := Tape{Tensors: []TensorSpec{{Shape: []int{n}, Values: values}, {Shape: []int{n}}}, Commands: []Command{{Op: "reshape", Inputs: []int{0}, Output: 1}}, Outputs: []int{1}}
	b.ReportAllocs()
	b.SetBytes(n * 4)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := tape.Validate(8 * n); err != nil {
			b.Fatal(err)
		}
	}
}

func TestTrainingTapeFiniteBitBoundaries(t *testing.T) {
	for _, bits := range []uint32{0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000, 0x80800000, 0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc00000, 0xffffffff} {
		tape := vectorTapeFixture()
		tape.Tensors[0].Values[0] = math.Float32frombits(bits)
		err := tape.Validate(1024)
		wantFinite := bits&0x7f800000 != 0x7f800000
		if (err == nil) != wantFinite {
			t.Fatalf("bits=%08x admission=%v finite=%t", bits, err, wantFinite)
		}
	}
}
