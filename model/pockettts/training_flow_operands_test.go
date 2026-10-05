package pockettts

import (
	"sync"
	"testing"
)

func TestTrainingFlowOperandArenaCopies(t *testing.T) {
	var a flowOperandArena
	if a.copy(nil) != nil {
		t.Fatal("nil operands")
	}
	for _, size := range []int{1, 4, 8, 9} {
		inputs := make([]int, size)
		for i := range inputs {
			inputs[i] = i + 1
		}
		got := a.copy(inputs)
		inputs[0] = 99
		if got[0] != 1 || len(got) != size {
			t.Fatal("source borrowed")
		}
		got[0] = 10
		grown := append(got, 20)
		grown[0] = 30
		if got[0] != 10 {
			t.Fatal("append alias")
		}
	}
	var saved [][]int
	for i := 0; i < 200; i++ {
		saved = append(saved, a.copy([]int{i + 1, 2, 3, 4, 5}))
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
func TestTrainingFlowOperandArenaGraphs(t *testing.T) {
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
	for i, c := range a.Tape.Commands {
		if cap(c.Inputs) != len(c.Inputs) {
			t.Fatal("operand capacity")
		}
		before := b.Tape.Commands[i].Inputs[0]
		c.Inputs[0] = -1
		if b.Tape.Commands[i].Inputs[0] != before {
			t.Fatal("cross-call alias")
		}
		c.Inputs[0] = before
	}
	values := interpretFlowTape(t, a.Tape)
	b.Tape.Commands[0].Inputs[0] = -1
	again := interpretFlowTape(t, a.Tape)
	for i, x := range values {
		assertSliceClose(t, "retained graph", again[i], x, 2e-6)
	}
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
				p.Tape.Commands[0].Inputs[0] = -1
			}
		}()
	}
	wg.Wait()
}
func TestTrainingFlowOperandArenaCommandSource(t *testing.T) {
	g := &flowTapeGraph{}
	source := []int{0, 1}
	id := g.command("vector", source, []int{2}, 0, 0, 0, 1, 0)
	source[0] = 99
	if id != 0 || g.tape.Commands[0].Inputs[0] != 0 || cap(g.tape.Commands[0].Inputs) != 2 {
		t.Fatal("command source ownership")
	}
}

func TestTrainingFlowOperandArenaSiblings(t *testing.T) {
	g := &flowTapeGraph{}
	inputs := []int{0, 1, 2, 3, 4}
	g.command("linear-backward", inputs, []int{1, 1}, 0, 0, 0, 0, 0)
	g.command("linear-backward", inputs, []int{1, 1}, 0, 1, 0, 0, 0)
	g.tape.Commands[0].Inputs[0] = 99
	if g.tape.Commands[1].Inputs[0] != 0 || inputs[0] != 0 {
		t.Fatal("sibling or caller alias")
	}
	grown := append(g.tape.Commands[0].Inputs, 77)
	grown[0] = 123
	if g.tape.Commands[0].Inputs[0] != 99 || g.tape.Commands[1].Inputs[0] != 0 {
		t.Fatal("append reaches sibling")
	}
}
