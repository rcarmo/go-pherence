package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"testing"
)

func TestTrainingFlowMetadataCounts(t *testing.T) {
	for _, device := range []bool{false, true} {
		for _, blocks := range []int{1, 2, 4} {
			m := tinyTrainableFlowHead()
			m.Blocks = m.Blocks[:1]
			for len(m.Blocks) < blocks {
				m.Blocks = append(m.Blocks, m.Blocks[0])
			}
			in := primaryFixture()
			_, err := m.buildFlowDualTrainingTapeSeed(in.Condition, []float32{.3, .8}, in.Input, 1, in.GV, in.GT, 64<<20, device, true, func(g *flowTapeGraph, out graphDual) (int, int) {
				gv, gt := g.constant(in.GV), g.constant(in.GT)
				c := *g
				c.shapeArena = flowShapeArena{} // independent append schedule
				c.operandArena = flowOperandArena{}
				c.tape.Tensors = append([]train.TensorSpec(nil), g.tape.Tensors...)
				c.tape.Commands = append([]train.Command(nil), g.tape.Commands...)
				c.adj = map[int]int{out.v: gv, out.t: gt}
				count := c.reverseCommandCount()
				before := len(c.tape.Commands)
				c.reverseIDs(out, gv, gt)
				if count != len(c.tape.Commands)-before {
					t.Fatal("incorrect reverse command count", count, len(c.tape.Commands)-before)
				}
				if before > flowForwardCommandCapacity(len(m.Time), blocks, device) {
					t.Fatal("forward estimate too small", device, blocks, before, flowForwardCommandCapacity(len(m.Time), blocks, device))
				}
				return gv, gt
			})
			if err != nil {
				t.Fatal(err)
			}
		}
	}
}
func TestTrainingFlowMetadataCapacityBounds(t *testing.T) {
	if flowForwardCommandCapacity(0, 0, false) != 51 || flowForwardCommandCapacity(1, 1, true) != 115 {
		t.Fatal("capacity formula")
	}
	if flowForwardCommandCapacity(100000, 100000, true) != 4096 {
		t.Fatal("capacity ceiling")
	}
	// Existing sufficient storage is reused and headers remain unchanged.
	g := &flowTapeGraph{tape: train.Tape{Commands: make([]train.Command, 0, 128)}, adj: map[int]int{}}
	g.reserveReverseMetadata()
	if cap(g.tape.Commands) != 128 || len(g.tape.Commands) != 0 {
		t.Fatal("unnecessary reserve")
	}
}

func TestTrainingFlowMetadataOwnership(t *testing.T) {
	g := &flowTapeGraph{tape: train.Tape{Tensors: []train.TensorSpec{{Shape: []int{1}, Values: []float32{1}}}, Commands: []train.Command{{Op: "reshape", Inputs: []int{0}, Output: 0}}, Outputs: []int{0}}, adj: map[int]int{0: 0}}
	commands := g.tape.Commands
	tensors := g.tape.Tensors
	adj := len(g.adj)
	if g.reverseCommandCount() != 0 || len(g.adj) != adj {
		t.Fatal("count mutated adjoints")
	}
	g.reserveReverseMetadata()
	if len(g.tape.Commands) != 1 || g.tape.Commands[0].Inputs[0] != 0 || len(g.tape.Tensors) != 1 || &g.tape.Tensors[0] != &tensors[0] || g.tape.Outputs[0] != 0 {
		t.Fatal("metadata changed")
	}
	g.tape.Commands[0].Output = 1
	if commands[0].Output != 0 {
		t.Fatal("command header alias")
	}
	// Reservation is capped even for a synthetic oversized AD schedule; final
	// Tape.Validate, not capacity, determines whether that schedule is admitted.
	g.nodes = make([]graphNode, 1100)
	for i := range g.nodes {
		g.nodes[i] = graphNode{op: "silu", input: [2]graphDual{{0, 0}}, output: graphDual{0, 0}}
	}
	g.reserveReverseMetadata()
	if cap(g.tape.Commands) != 4096 {
		t.Fatal("reservation unbounded")
	}
}
