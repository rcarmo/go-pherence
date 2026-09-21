package graph

import "testing"

func TestPlanDuplicateInputReleasesOnce(t *testing.T) {
	g := New("repeat-input")
	x := g.AddValue("x", Shape{8}, F32, false)
	y := g.AddValue("y", Shape{8}, F32, false)
	a := g.AddValue("a", Shape{8}, F32, false)
	b := g.AddValue("b", Shape{8}, F32, false)
	c := g.AddValue("c", Shape{8}, F32, false)
	g.AddNode("input", OpInput, nil, []ValueID{x}, nil)
	g.AddNode("square", OpMul, []ValueID{x, x}, []ValueID{y}, nil)
	g.AddNode("a", OpConst, nil, []ValueID{a}, nil)
	g.AddNode("b", OpConst, nil, []ValueID{b}, nil)
	g.AddNode("consume", OpAdd, []ValueID{a, b}, []ValueID{c}, nil)
	p, err := BuildPlan(g)
	if err != nil {
		t.Fatal(err)
	}
	if p.Steps[2].OutputBufs[0] == p.Steps[3].OutputBufs[0] {
		t.Fatal("simultaneously live outputs alias one freed buffer")
	}
}
func TestGraphInvalidIDsShapesAndWorkspace(t *testing.T) {
	max := int(^uint(0) >> 1)
	for _, shape := range []Shape{{-1}, {max, 2}, {max/4 + 1}, {0, -1}} {
		g := New("bad")
		g.AddValue("v", shape, F32, false)
		if _, err := BuildPlan(g); err == nil {
			t.Fatal("accepted", shape)
		}
	}
	for _, output := range []bool{false, true} {
		g := New("bad")
		x := g.AddValue("x", Shape{1}, F32, true)
		inputs, outputs := []ValueID{-1}, []ValueID{x}
		if output {
			inputs, outputs = []ValueID{x}, []ValueID{-1}
		}
		g.AddNode("bad", OpAdd, inputs, outputs, nil)
		if _, err := BuildPlan(g); err == nil {
			t.Fatal("negative id accepted")
		}
	}
	g := New("huge")
	for i := 0; i < 3; i++ {
		v := g.AddValue("large", Shape{max / 8}, F32, false)
		g.AddNode("input", OpInput, nil, []ValueID{v}, nil)
	}
	if _, err := BuildPlan(g); err == nil {
		t.Fatal("workspace sum overflow accepted")
	}
	g = New("redefine")
	x := g.AddValue("x", Shape{1}, F32, false)
	g.AddNode("input", OpInput, nil, []ValueID{x}, nil)
	g.AddNode("rewrite", OpAdd, []ValueID{x}, []ValueID{x}, nil)
	if _, err := BuildPlan(g); err == nil {
		t.Fatal("transient redefinition accepted")
	}
	var nilGraph *Graph
	if err := nilGraph.Validate(); err == nil {
		t.Fatal("nil graph accepted")
	}
}
func TestAddValueOwnsShape(t *testing.T) {
	g := New("shape")
	s := Shape{2}
	id := g.AddValue("x", s, F32, false)
	s[0] = -1
	if g.Value(id).Shape[0] != 2 {
		t.Fatal("shape aliases caller")
	}
}

func TestPlanViewChainsRetainBackingStorage(t *testing.T) {
	for _, op := range []OpKind{OpView, OpReshape, OpSlice, OpTranspose, OpContiguous} {
		t.Run(string(op), func(t *testing.T) {
			g := New("view lifetime")
			x := g.AddValue("x", Shape{8}, F32, false)
			v := g.AddValue("view", Shape{8}, F32, false)
			v2 := g.AddValue("view2", Shape{8}, F32, false)
			other := g.AddValue("other", Shape{8}, F32, false)
			result := g.AddValue("result", Shape{8}, F32, false)
			g.AddNode("input", OpInput, nil, []ValueID{x}, nil)
			g.AddNode("view", op, []ValueID{x}, []ValueID{v}, nil)
			g.AddNode("view2", OpReshape, []ValueID{v}, []ValueID{v2}, nil)
			g.AddNode("other", OpConst, nil, []ValueID{other}, nil)
			g.AddNode("consumer", OpAdd, []ValueID{v2, other}, []ValueID{result}, nil)
			p, err := BuildPlan(g)
			if err != nil {
				t.Fatal(err)
			}
			backing := p.Steps[0].OutputBufs[0]
			if p.LastUse[x] != 4 || p.LastUse[v] != 4 || p.Steps[3].OutputBufs[0] == backing || p.Steps[4].OutputBufs[0] == backing {
				t.Fatalf("view backing reused prematurely: %+v", p)
			}
		})
	}
}
func TestPlanTerminalViewRetainsInput(t *testing.T) {
	g := New("terminal view")
	x := g.AddValue("x", Shape{8}, F32, false)
	v := g.AddValue("returned view", Shape{8}, F32, false)
	other := g.AddValue("other", Shape{8}, F32, false)
	g.AddNode("input", OpInput, nil, []ValueID{x}, nil)
	g.AddNode("view", OpView, []ValueID{x}, []ValueID{v}, nil)
	g.AddNode("other", OpConst, nil, []ValueID{other}, nil)
	p, err := BuildPlan(g)
	if err != nil {
		t.Fatal(err)
	}
	if p.LastUse[x] != len(g.Nodes) || p.Steps[0].OutputBufs[0] == p.Steps[2].OutputBufs[0] {
		t.Fatal("returned view overwritten")
	}
}
