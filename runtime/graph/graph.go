// Package graph defines a small planned execution IR for model inference.
//
// It is intentionally backend-neutral: model packages build Graphs, then backends
// lower Plans to concrete execution (Go/RVV, GGML, ORT, Vulkan, libllama, ...).
package graph

import "fmt"

// DType is the logical element type of a graph value.
type DType string

const (
	F32  DType = "f32"
	F16  DType = "f16"
	BF16 DType = "bf16"
	I32  DType = "i32"
	I64  DType = "i64"
	Q2K  DType = "q2_k"
	Q3K  DType = "q3_k"
	Q6K  DType = "q6_k"
	Q8K  DType = "q8_k"
)

// Shape is a row-major logical tensor shape.
type Shape []int

// Numel returns the element count; 0 means dynamic/invalid.
func (s Shape) Numel() int {
	if len(s) == 0 {
		return 0
	}
	n := 1
	for _, d := range s {
		if d <= 0 {
			return 0
		}
		n *= d
	}
	return n
}

func (s Shape) String() string { return fmt.Sprint([]int(s)) }

// ValueID identifies a logical graph value.
type ValueID int

// OpKind identifies a backend-lowerable operation.
type OpKind string

const (
	OpInput     OpKind = "input"
	OpConst     OpKind = "const"
	OpEmbedding OpKind = "embedding"
	OpRMSNorm   OpKind = "rms_norm"
	OpMatMul    OpKind = "matmul"
	OpRoPE      OpKind = "rope"
	OpAttention OpKind = "attention"
	OpSoftmax   OpKind = "softmax"
	OpSiLU      OpKind = "silu"
	OpMul       OpKind = "mul"
	OpAdd       OpKind = "add"
	OpCopy      OpKind = "copy"
	OpView      OpKind = "view"
	OpReshape   OpKind = "reshape"
	OpKVRead    OpKind = "kv_read"
	OpKVWrite   OpKind = "kv_write"
	OpSample    OpKind = "sample"
)

// Value describes a tensor flowing through the graph.
type Value struct {
	ID         ValueID
	Name       string
	Shape      Shape
	DType      DType
	Persistent bool // model weights, KV cache, or runtime-owned persistent buffers
}

// Node is one operation in topological order.
type Node struct {
	ID      int
	Name    string
	Op      OpKind
	Inputs  []ValueID
	Outputs []ValueID
	Attrs   map[string]any
}

// Graph is a model-level logical execution graph.
type Graph struct {
	Name   string
	Values []Value
	Nodes  []Node
}

// New creates an empty graph.
func New(name string) *Graph { return &Graph{Name: name} }

// AddValue appends a value and returns its ID.
func (g *Graph) AddValue(name string, shape Shape, dtype DType, persistent bool) ValueID {
	id := ValueID(len(g.Values))
	g.Values = append(g.Values, Value{ID: id, Name: name, Shape: append(Shape(nil), shape...), DType: dtype, Persistent: persistent})
	return id
}

// AddNode appends a topologically ordered node.
func (g *Graph) AddNode(name string, op OpKind, inputs, outputs []ValueID, attrs map[string]any) int {
	id := len(g.Nodes)
	if attrs == nil {
		attrs = map[string]any{}
	}
	g.Nodes = append(g.Nodes, Node{ID: id, Name: name, Op: op, Inputs: append([]ValueID(nil), inputs...), Outputs: append([]ValueID(nil), outputs...), Attrs: attrs})
	return id
}

// Value returns the value descriptor for id.
func (g *Graph) Value(id ValueID) (Value, bool) {
	if id < 0 || int(id) >= len(g.Values) {
		return Value{}, false
	}
	return g.Values[id], true
}

// Validate checks basic graph consistency.
func (g *Graph) Validate() error {
	defined := make([]bool, len(g.Values))
	for _, v := range g.Values {
		if v.Persistent {
			// Inputs/constants are considered externally defined until overwritten.
			defined[v.ID] = true
		}
	}
	for _, n := range g.Nodes {
		if n.Name == "" {
			return fmt.Errorf("node %d has empty name", n.ID)
		}
		for _, in := range n.Inputs {
			if in < 0 || int(in) >= len(g.Values) {
				return fmt.Errorf("node %s input %d out of range", n.Name, in)
			}
			if !defined[in] {
				return fmt.Errorf("node %s input %s used before definition", n.Name, g.Values[in].Name)
			}
		}
		for _, out := range n.Outputs {
			if out < 0 || int(out) >= len(g.Values) {
				return fmt.Errorf("node %s output %d out of range", n.Name, out)
			}
			defined[out] = true
		}
	}
	return nil
}
