package graph

import "fmt"

// TensorRef is an executor-owned view of a logical value.
type TensorRef struct {
	Value ValueID
	Buf   BufferID
}

// Executor lowers and executes a Plan on a backend.
type Executor interface {
	Prepare(*Plan) error
	Run(*Plan) error
}

// NodeFunc executes one planned node. It is useful for simple/native backends.
type NodeFunc func(step Step) error

// FuncExecutor executes plans by dispatching each node to registered functions.
type FuncExecutor struct {
	Ops map[OpKind]NodeFunc
}

func (e *FuncExecutor) Prepare(*Plan) error { return nil }

func (e *FuncExecutor) Run(p *Plan) error {
	if p == nil {
		return fmt.Errorf("nil plan")
	}
	for _, st := range p.Steps {
		fn := e.Ops[st.Node.Op]
		if fn == nil {
			return fmt.Errorf("no executor for op %s (%s)", st.Node.Op, st.Node.Name)
		}
		if err := fn(st); err != nil {
			return fmt.Errorf("%s: %w", st.Node.Name, err)
		}
	}
	return nil
}
