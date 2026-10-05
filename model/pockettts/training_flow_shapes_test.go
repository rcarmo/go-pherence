package pockettts

import (
	"sync"
	"testing"
)

func TestTrainingFlowShapeArenaCopies(t *testing.T) {
	var a flowShapeArena
	if a.copy(nil) != nil {
		t.Fatal("nil shape")
	}
	for _, size := range []int{1, 2, 3, 4} {
		shape := make([]int, size)
		for i := range shape {
			shape[i] = i + 1
		}
		got := a.copy(shape)
		shape[0] = 99
		if got[0] != 1 || len(got) != size {
			t.Fatal("source borrowed")
		}
		got[0] = 10
		grown := append(got, 20)
		grown[0] = 30
		if got[0] != 10 {
			t.Fatal("append not isolated")
		}
	}
	var saved [][]int
	for i := 0; i < 200; i++ {
		shape := []int{i + 1, 2, 3}
		saved = append(saved, a.copy(shape))
	}
	for i, x := range saved {
		if x[0] != i+1 || cap(x) != len(x) {
			t.Fatal("chunk overwrite", i)
		}
	}
	saved[0][0] = 123
	if saved[1][0] != 2 {
		t.Fatal("neighbour alias")
	}
}
func TestTrainingFlowShapeArenaGraphs(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := primaryFixture()
	build := func() FlowDualTrainingTape {
		p, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, []float32{.3, .8}, in.Input, 1, in.GV, in.GT, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	a, b := build(), build()
	for i, s := range a.Tape.Tensors {
		if cap(s.Shape) != len(s.Shape) {
			t.Fatal("unbounded shape slice")
		}
		before := b.Tape.Tensors[i].Shape[0]
		s.Shape[0] = 99
		if b.Tape.Tensors[i].Shape[0] != before {
			t.Fatal("cross-call shape alias")
		}
		s.Shape[0] = before
	}
	values := interpretFlowTape(t, a.Tape)
	b.Tape.Tensors[0].Shape[0] = 99
	again := interpretFlowTape(t, a.Tape)
	for i, x := range values {
		assertSliceClose(t, "retained graph", again[i], x, 2e-6)
	}
	// Independent builders may mutate their shape metadata concurrently.
	var wg sync.WaitGroup
	for worker := 0; worker < 4; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 5; i++ {
				p, err := m.BuildFlowDeviceTimeTrainingTape(in.Condition, []float32{.3, .8}, in.Input, 1, in.GV, in.GT, 64<<20)
				if err != nil {
					t.Error(err)
					return
				}
				p.Tape.Tensors[0].Shape[0] = 77
			}
		}()
	}
	wg.Wait()
}
