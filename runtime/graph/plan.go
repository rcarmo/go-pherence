package graph

import "fmt"

// BufferID identifies a planned transient buffer slot.
type BufferID int

// Step is a planned node plus its transient buffer assignment.
type Step struct {
	Node       Node
	InputBufs  []BufferID // -1 means external/persistent value
	OutputBufs []BufferID // -1 means external/persistent value
}

// Buffer is a reusable transient allocation slot.
type Buffer struct {
	ID    BufferID
	Name  string
	Shape Shape
	DType DType
	Bytes int
}

// Plan is a topologically ordered, lifetime-aware execution plan.
type Plan struct {
	Graph   *Graph
	Steps   []Step
	Buffers []Buffer
	LastUse map[ValueID]int
}

// DTypeSize returns an approximate byte size for physical planning.
func DTypeSize(t DType) int {
	switch t {
	case F32, I32:
		return 4
	case F16, BF16:
		return 2
	case I64:
		return 8
	case Q2K, Q3K, Q6K, Q8K, Q4G:
		// Quantized weights are persistent. Transients should not use this path.
		return 1
	default:
		return 4
	}
}

func valueBytes(v Value) int {
	n := v.Shape.Numel()
	if n <= 0 {
		return 0
	}
	return n * DTypeSize(v.DType)
}

// BuildPlan validates g and assigns reusable transient buffers.
func BuildPlan(g *Graph) (*Plan, error) {
	if g == nil {
		return nil, fmt.Errorf("nil graph")
	}
	if err := g.Validate(); err != nil {
		return nil, err
	}
	lastUse := make(map[ValueID]int)
	for i, n := range g.Nodes {
		for _, in := range n.Inputs {
			lastUse[in] = i
		}
	}

	valueBuf := make(map[ValueID]BufferID)
	free := []BufferID{}
	buffers := []Buffer{}
	steps := make([]Step, 0, len(g.Nodes))

	alloc := func(v Value) BufferID {
		if v.Persistent {
			return -1
		}
		need := valueBytes(v)
		// First-fit reuse: same dtype and enough bytes.
		for i, bid := range free {
			b := buffers[bid]
			if b.DType == v.DType && b.Bytes >= need {
				free = append(free[:i], free[i+1:]...)
				return bid
			}
		}
		bid := BufferID(len(buffers))
		buffers = append(buffers, Buffer{ID: bid, Name: v.Name, Shape: append(Shape(nil), v.Shape...), DType: v.DType, Bytes: need})
		return bid
	}

	for i, n := range g.Nodes {
		step := Step{Node: n}
		for _, in := range n.Inputs {
			if g.Values[in].Persistent {
				step.InputBufs = append(step.InputBufs, -1)
			} else {
				step.InputBufs = append(step.InputBufs, valueBuf[in])
			}
		}
		for _, out := range n.Outputs {
			v := g.Values[out]
			bid := alloc(v)
			valueBuf[out] = bid
			step.OutputBufs = append(step.OutputBufs, bid)
		}
		steps = append(steps, step)

		// Release inputs whose lifetime ends here.
		for _, in := range n.Inputs {
			if g.Values[in].Persistent {
				continue
			}
			if lastUse[in] == i {
				free = append(free, valueBuf[in])
			}
		}
	}
	return &Plan{Graph: g, Steps: steps, Buffers: buffers, LastUse: lastUse}, nil
}

// WorkspaceBytes returns the total transient workspace bytes.
func (p *Plan) WorkspaceBytes() int {
	if p == nil {
		return 0
	}
	var n int
	for _, b := range p.Buffers {
		n += b.Bytes
	}
	return n
}
