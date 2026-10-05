package training

import (
	"fmt"
	"sync"
	"testing"
)

var pruneAllocationSink Tape
var pruneMappingSink []int

// A deterministic vector chain with a dead independent branch. This exercises
// metadata construction and admission without a driver or numerical execution.
func pruneAllocationFixture(commands int) Tape {
	t := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}, {Shape: []int{2}, Values: []float32{0, 0}}}}
	input := 0
	for i := 0; i < commands; i++ {
		id := len(t.Tensors)
		t.Tensors = append(t.Tensors, TensorSpec{Shape: []int{2}})
		t.Commands = append(t.Commands, Command{Op: "vector", Inputs: []int{input, 1}, Output: id, Scale: 1})
		input = id
	}
	dead := len(t.Tensors)
	t.Tensors = append(t.Tensors, TensorSpec{Shape: []int{2}})
	t.Commands = append(t.Commands, Command{Op: "vector", Inputs: []int{0, 1}, Output: dead, Scale: 2})
	t.Outputs = []int{input, dead}
	return t
}
func BenchmarkTrainingTapePruneAllocation(b *testing.B) {
	for _, n := range []int{1, 128, 1024} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			source := pruneAllocationFixture(n)
			roots := []int{source.Outputs[0]}
			if _, _, err := PruneTape(source, roots, 64<<20); err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				t, m, err := PruneTape(source, roots, 64<<20)
				if err != nil {
					b.Fatal(err)
				}
				pruneAllocationSink, pruneMappingSink = t, m
			}
		})
	}
}
func TestTrainingTapePruneLargeOrder(t *testing.T) {
	for _, n := range []int{1, 128, 1024} {
		source := pruneAllocationFixture(n)
		result, mapping, err := PruneTape(source, []int{source.Outputs[0], source.Outputs[0]}, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Commands) != n || len(result.Tensors) != n+2 || mapping[len(source.Tensors)-1] != -1 || len(result.Outputs) != 2 || result.Outputs[0] != result.Outputs[1] || cap(result.Commands) != n || cap(result.Tensors) != n+2 || cap(result.Outputs) != 2 {
			t.Fatal("closure", n)
		}
		for i, c := range result.Commands {
			if c.Output != i+2 || c.Inputs[0] != (func() int {
				if i == 0 {
					return 0
				}
				return i + 1
			})() || c.Inputs[1] != 1 || c.Scale != 1 || c.Op != "vector" {
				t.Fatal("order", i)
			}
		}
		// Owned metadata may mutate; constants remain deliberately read-only borrowed.
		result.Commands[0].Inputs[0] = 99
		result.Tensors[0].Shape[0] = 99
		result.Outputs[0] = 99
		if source.Commands[0].Inputs[0] != 0 || source.Tensors[0].Shape[0] != 2 || source.Outputs[0] != n+1 {
			t.Fatal("metadata alias")
		}
		if &result.Tensors[0].Values[0] != &source.Tensors[0].Values[0] {
			t.Fatal("value contract changed")
		}
	}
}

func TestTrainingTapePruneConstantMetadata(t *testing.T) {
	source := Tape{Tensors: []TensorSpec{{Shape: []int{2}, Values: []float32{1, 2}}}, Outputs: []int{0}}
	result, mapping, err := PruneTape(source, []int{0, 0}, 8)
	if err != nil {
		t.Fatal(err)
	}
	if result.Commands != nil || cap(result.Tensors) != 1 || cap(result.Outputs) != 2 || len(mapping) != 1 || mapping[0] != 0 {
		t.Fatal("constant metadata contract")
	}
}

func TestTrainingTapePruneMetadataIsolation(t *testing.T) {
	source := pruneAllocationFixture(128)
	a, _, err := PruneTape(source, source.Outputs, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	b, _, err := PruneTape(source, source.Outputs, 64<<20)
	if err != nil {
		t.Fatal(err)
	}
	for i, s := range a.Tensors {
		if cap(s.Shape) != len(s.Shape) {
			t.Fatal("shape append can escape", i)
		}
		before := b.Tensors[i].Shape[0]
		s.Shape[0] = 99
		if b.Tensors[i].Shape[0] != before || source.Tensors[i].Shape[0] != before {
			t.Fatal("shape alias")
		}
		grown := append(s.Shape, 1, 2, 3)
		grown[0] = 100
		if s.Shape[0] != 99 {
			t.Fatal("shape append aliased")
		}
	}
	for i, c := range a.Commands {
		if cap(c.Inputs) != len(c.Inputs) {
			t.Fatal("operand append can escape", i)
		}
		before := b.Commands[i].Inputs[0]
		c.Inputs[0] = 99
		if b.Commands[i].Inputs[0] != before || source.Commands[i].Inputs[0] != before {
			t.Fatal("operand alias")
		}
		grown := append(c.Inputs, 1, 2, 3)
		grown[0] = 100
		if c.Inputs[0] != 99 {
			t.Fatal("operand append aliased")
		}
	}
	// Parallel independent prune results may mutate metadata without cross-talk.
	var wg sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				result, _, err := PruneTape(source, source.Outputs, 64<<20)
				if err != nil {
					t.Error(err)
					return
				}
				result.Tensors[0].Shape[0] = 77
				result.Commands[0].Inputs[0] = 77
			}
		}()
	}
	wg.Wait()
	if source.Tensors[0].Shape[0] != 2 || source.Commands[0].Inputs[0] != 0 {
		t.Fatal("source mutated")
	}
}
func TestTrainingTapePruneMetadataArena(t *testing.T) {
	arena := pruneMetadataArena{unassigned: 1000}
	first := arena.take(3)
	second := arena.take(4)
	first[0] = 1
	second[0] = 2
	if len(first) != 3 || cap(first) != 3 || len(second) != 4 || cap(second) != 4 || first[0] != 1 {
		t.Fatal("arena partitions")
	}
	for i := 0; i < 100; i++ {
		x := arena.take(7)
		if len(x) != 7 || cap(x) != 7 {
			t.Fatal("chunk boundary")
		}
	}
	// Current admitted metadata is small, but a defensive larger request still
	// has isolated owned storage instead of panicking.
	large := arena.take(257)
	if len(large) != 257 || cap(large) != 257 {
		t.Fatal("large metadata")
	}
}
