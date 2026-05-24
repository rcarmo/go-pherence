package graph

import "testing"

func TestBuildPlanReusesTransientBuffers(t *testing.T) {
	g := New("toy")
	x := g.AddValue("x", Shape{4}, F32, true)
	w := g.AddValue("w", Shape{4, 4}, F32, true)
	a := g.AddValue("a", Shape{4}, F32, false)
	b := g.AddValue("b", Shape{4}, F32, false)
	c := g.AddValue("c", Shape{4}, F32, false)
	g.AddNode("a=matmul", OpMatMul, []ValueID{w, x}, []ValueID{a}, nil)
	g.AddNode("b=silu", OpSiLU, []ValueID{a}, []ValueID{b}, nil)
	g.AddNode("c=add", OpAdd, []ValueID{x, b}, []ValueID{c}, nil)

	p, err := BuildPlan(g)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Steps) != 3 {
		t.Fatalf("steps=%d", len(p.Steps))
	}
	if len(p.Buffers) > 2 {
		t.Fatalf("expected reuse with <=2 buffers, got %d", len(p.Buffers))
	}
	if p.WorkspaceBytes() <= 0 {
		t.Fatalf("workspace bytes = %d", p.WorkspaceBytes())
	}
}

func TestValidateCatchesUseBeforeDef(t *testing.T) {
	g := New("bad")
	x := g.AddValue("x", Shape{4}, F32, false)
	y := g.AddValue("y", Shape{4}, F32, false)
	g.AddNode("bad", OpAdd, []ValueID{x}, []ValueID{y}, nil)
	if err := g.Validate(); err == nil {
		t.Fatal("expected validation error")
	}
}
