package training

import (
	"math"
	"math/rand"
	"testing"
)

// Independent chronological scalar evaluation compares pooled storage with
// separate SSA values. NaN-prefilled slots expose unwritten/stale elements;
// no hashes or byte-equality gates establish numerical parity.
func TestTrainingTapeStorageIndependentReference(t *testing.T) {
	random := rand.New(rand.NewSource(37))
	for trial := 0; trial < 100; trial++ {
		n := 1 + random.Intn(8)
		tape := Tape{}
		for j := 0; j < 3; j++ {
			values := make([]float32, n)
			for i := range values {
				values[i] = float32(random.Intn(17)-8) / 8
			}
			tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{n}, Values: values})
		}
		for k := 0; k < 30; k++ {
			id := len(tape.Tensors)
			a, b := random.Intn(id), random.Intn(id)
			tape.Tensors = append(tape.Tensors, TensorSpec{Shape: []int{n}})
			c := Command{Op: "vector", Inputs: []int{a, b}, Output: id, Scale: float32(random.Intn(5)-2) / 4}
			if k%7 == 0 {
				c = Command{Op: "reshape", Inputs: []int{a}, Output: id}
			}
			tape.Commands = append(tape.Commands, c)
		}
		tape.Outputs = []int{len(tape.Tensors) - 1, 3 + random.Intn(30), random.Intn(3)}
		p, err := PlanTapeStorage(tape, 1<<20)
		if err != nil {
			t.Fatal(err)
		}
		slots := make([][]float32, len(p.Elements))
		for i, n := range p.Elements {
			slots[i] = make([]float32, n)
			for k := range slots[i] {
				slots[i][k] = float32(math.NaN())
			}
		}
		reference := make([][]float32, len(tape.Tensors))
		for i, s := range tape.Tensors {
			if s.Values != nil {
				reference[i] = append([]float32(nil), s.Values...)
				copy(slots[p.Slots[i]], s.Values)
			}
		}
		for _, c := range tape.Commands {
			if c.Op == "reshape" {
				reference[c.Output] = append([]float32(nil), reference[c.Inputs[0]]...)
				if p.Slots[c.Output] != p.Slots[c.Inputs[0]] {
					t.Fatal("reshape slot")
				}
				continue
			}
			dst := slots[p.Slots[c.Output]]
			reference[c.Output] = make([]float32, n)
			for _, id := range c.Inputs {
				if p.Slots[c.Output] == p.Slots[id] {
					t.Fatal("dispatch input clobber", trial, c)
				}
			}
			for i := range dst {
				reference[c.Output][i] = reference[c.Inputs[0]][i] + c.Scale*reference[c.Inputs[1]][i]
				dst[i] = slots[p.Slots[c.Inputs[0]]][i] + c.Scale*slots[p.Slots[c.Inputs[1]]][i]
			}
		}
		for _, id := range tape.Outputs {
			for i, want := range reference[id] {
				got := slots[p.Slots[id]][i]
				if math.IsNaN(float64(got)) || math.Abs(float64(got-want)) > 1e-7+1e-6*math.Abs(float64(want)) {
					t.Fatal("SSA reference", trial, id, i, got, want)
				}
			}
		}
	}
}
