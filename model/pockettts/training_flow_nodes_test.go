package pockettts

import (
	"reflect"
	"testing"
)

func TestTrainingFlowNodeInlineInputs(t *testing.T) {
	m := tinyTrainableFlowHead()
	in := primaryFixture()
	for _, device := range []bool{false, true} {
		seen := map[string]bool{}
		_, err := m.buildFlowDualTrainingTapeSeed(in.Condition, []float32{.3, .8}, in.Input, 1, in.GV, in.GT, 64<<20, device, true, func(g *flowTapeGraph, out graphDual) (int, int) {
			for i, node := range g.nodes {
				seen[node.op] = true
				slots := 1
				switch node.op {
				case "mod", "add":
					slots = 2
				case "linear", "silu", "norm", "slice":
				default:
					t.Fatal("unexpected flow node", node.op)
				}
				for _, x := range node.input[:slots] {
					if x.v < 0 || x.v >= len(g.tape.Tensors) || x.t < 0 || x.t >= len(g.tape.Tensors) {
						t.Fatal("invalid operand IDs", node.op)
					}
				}
				if slots == 1 && node.input[1] != (graphDual{}) {
					t.Fatal("unused unary slot", node.op)
				}
				// A node copied for reverse/counting has value-owned operand slots.
				// Editing that copy cannot change the retained forward schedule.
				saved := g.nodes[i]
				node.input[0].v = -1
				node.input[1].t = -1
				if !reflect.DeepEqual(g.nodes[i], saved) {
					t.Fatal("copied node inputs alias", node.op)
				}
			}
			// Owning an independent node-header slice must also own the inline inputs.
			saved := g.nodes[0]
			copyNodes := append([]graphNode(nil), g.nodes...)
			copyNodes[0].input[0].v = -1
			if !reflect.DeepEqual(g.nodes[0], saved) {
				t.Fatal("cloned schedule inputs alias")
			}
			return g.constant(in.GV), g.constant(in.GT)
		})
		if err != nil {
			t.Fatal(err)
		}
		for _, op := range []string{"linear", "silu", "norm", "mod", "add", "slice"} {
			if !seen[op] {
				t.Fatal("unexercised node", op, device)
			}
		}
	}
}
