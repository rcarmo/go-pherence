// Package graph defines a backend-neutral planned execution IR for model inference.
//
// # Backend lowering targets
//
// The graph is designed so that the same IR can be lowered to multiple execution
// backends without changing the model-build code:
//
//   - GGML/CPU   — lower MatMul/RMSNorm/SiLU/Add/Mul to ggml_graph ops;
//                  wrap remaining decode ops in Go shims
//   - Vulkan     — lower compute-heavy subgraphs to SPIR-V kernels
//   - SpacemiT   — lower supported op islands to ORT/TCM execution providers
//   - MLX        — lower to mlx_array + mlx_stream C API (Apple Silicon only);
//                  StreamHint on each Node guides CPU vs Metal stream assignment;
//                  Q4G dtype maps to mlx_quantize(bits=4,group_size=64);
//                  Transpose/Concat/Slice map directly to mlx_transpose/mlx_concatenate/mlx_slice;
//                  Contiguous forces mlx_contiguous() before non-contiguous-aware ops;
//                  Scale maps to mlx_multiply(scalar) with a scalar Const input;
//                  ReduceSum/ReduceMean map to mlx_sum/mlx_mean with axis Attrs;
//                  GELU maps to mlx_gelu or mlx_gelu_approx per the Attrs["approx"] flag
//
// Op lowering coverage by backend:
//
//   Op            | GGML | Vulkan | SpacemiT | MLX
//   --------------|------|--------|----------|----
//   Embedding     |  ✓   |   -    |    -     |  ✓
//   RMSNorm       |  ✓   |   -    |    ✓     |  ✓
//   MatMul        |  ✓   |   ✓    |    ✓     |  ✓
//   RoPE          |  ✓   |   -    |    -     |  ✓
//   Attention     |  ✓   |   -    |    -     |  ✓
//   Softmax       |  ✓   |   -    |    ✓     |  ✓
//   SiLU          |  ✓   |   -    |    ✓     |  ✓
//   GELU          |  -   |   -    |    ✓     |  ✓
//   Mul           |  ✓   |   ✓    |    ✓     |  ✓
//   Add           |  ✓   |   ✓    |    ✓     |  ✓
//   Scale         |  ✓   |   -    |    -     |  ✓
//   Transpose     |  ✓   |   -    |    -     |  ✓
//   Concat        |  ✓   |   -    |    -     |  ✓
//   Slice         |  ✓   |   -    |    -     |  ✓
//   ReduceSum     |  ✓   |   -    |    -     |  ✓
//   ReduceMean    |  ✓   |   -    |    -     |  ✓
//   Contiguous    |  -   |   -    |    -     |  ✓
//   View/Reshape  |  ✓   |   -    |    -     |  ✓
//   KVRead/Write  |  -   |   -    |    -     |  -  (Go shim in all backends)
//   Sample        |  -   |   -    |    -     |  -  (Go shim in all backends)
package graph

import (
	"fmt"
	"strings"
)

// DType is the element type of a tensor value.
type DType string

const (
	// Floating point
	F32 DType = "f32"
	F16 DType = "f16"
	BF16 DType = "bf16"
	// Integer
	I32 DType = "i32"
	I64 DType = "i64"
	// GGML block quantisation (weight tensors only)
	Q2K DType = "q2_k"
	Q3K DType = "q3_k"
	Q6K DType = "q6_k"
	Q8K DType = "q8_k"
	// MLX / generic group quantisation (weight tensors only)
	// Q4G maps to mlx_quantize(bits=4, group_size=64).
	// Use Attrs["group_size"] on the owning MatMul node to override.
	Q4G DType = "q4_g"
)

// Shape is an ordered list of dimension sizes.
type Shape []int

func (s Shape) Numel() int {
	n := 1
	for _, d := range s {
		if d == 0 {
			return 0
		}
		n *= d
	}
	return n
}

func (s Shape) String() string {
	parts := make([]string, len(s))
	for i, d := range s {
		parts[i] = fmt.Sprintf("%d", d)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

// ValueID is a stable index into Graph.Values.
type ValueID int

// Value is a tensor descriptor (shape + dtype + lifetime).
type Value struct {
	ID         ValueID
	Name       string
	Shape      Shape
	DType      DType
	// Persistent marks weights, KV cache, and other buffers that live across
	// multiple graph executions and must not be reclaimed by the planner.
	Persistent bool
}

// OpKind names a compute operation.
type OpKind string

const (
	// Data sources
	OpInput     OpKind = "input"      // runtime input tensor (token ids, embeddings)
	OpConst     OpKind = "const"      // compile-time constant scalar or tensor

	// Memory ops
	OpEmbedding OpKind = "embedding"  // table lookup; Attrs: none
	OpView      OpKind = "view"       // zero-copy view; Attrs: "shape" []int
	OpReshape   OpKind = "reshape"    // reshape; Attrs: "shape" []int
	OpCopy      OpKind = "copy"       // explicit copy (e.g. device transfer)
	OpContiguous OpKind = "contiguous" // force contiguous layout (MLX lazy-eval materialization)
	OpTranspose OpKind = "transpose"  // permute dims; Attrs: "axes" []int (nil = last two dims)
	OpConcat    OpKind = "concat"     // concatenate along axis; Attrs: "axis" int
	OpSlice     OpKind = "slice"      // slice tensor; Attrs: "starts" []int, "stops" []int, "strides" []int

	// Elementwise / pointwise
	OpMul       OpKind = "mul"        // elementwise multiply
	OpAdd       OpKind = "add"        // elementwise add
	OpScale     OpKind = "scale"      // scalar multiply; second input is scalar Const (or Attrs["scale"] float32)
	OpSiLU      OpKind = "silu"       // SiLU activation
	OpGELU      OpKind = "gelu"       // GELU activation; Attrs: "approx" bool (tanh approx)
	OpSoftmax   OpKind = "softmax"    // softmax; Attrs: "axis" int (default -1)

	// Reduction
	OpReduceSum  OpKind = "reduce_sum"  // sum reduction; Attrs: "axes" []int, "keepdims" bool
	OpReduceMean OpKind = "reduce_mean" // mean reduction; Attrs: "axes" []int, "keepdims" bool

	// Linear algebra
	OpMatMul    OpKind = "matmul"     // matrix multiply; Attrs: "transA" bool, "transB" bool, "group_size" int (Q4G)
	OpRMSNorm   OpKind = "rms_norm"   // RMS normalisation; Attrs: "eps" float32
	OpRoPE      OpKind = "rope"       // rotary positional embedding; Attrs: "base" float32, "dims" int
	OpAttention OpKind = "attention"  // scaled dot-product attention; Attrs: "scale" float32, "causal" bool
	OpGather    OpKind = "gather"     // gather rows; generalisation of embedding; Attrs: "axis" int

	// KV cache (executor-managed; not lowered to hardware ops)
	OpKVRead    OpKind = "kv_read"    // read from persistent KV cache; Attrs: "layer" int, "kind" "k"|"v"
	OpKVWrite   OpKind = "kv_write"   // write to persistent KV cache; Attrs: "layer" int, "kind" "k"|"v", "pos" int

	// Output
	OpSample    OpKind = "sample"     // sample next token; Attrs: "temperature" float32, "top_k" int
)

// StreamKind hints which execution stream/device an executor should use for a node.
// Executors that don't support streams ignore this field.
type StreamKind int

const (
	StreamAny StreamKind = iota // executor chooses (default)
	StreamCPU                   // prefer CPU stream (MLX: mlx_default_cpu_stream)
	StreamGPU                   // prefer GPU/Metal stream (MLX: mlx_default_gpu_stream)
)

// Node is a single compute op in the graph.
type Node struct {
	ID      int
	Name    string
	Op      OpKind
	Inputs  []ValueID
	Outputs []ValueID
	Attrs   map[string]any
	// StreamHint guides backend executors that support multiple execution streams
	// (e.g. MLX CPU vs Metal stream, Vulkan queue family).
	// Executors that do not support streams must ignore this field.
	StreamHint StreamKind
}

// Graph is an ordered DAG of compute nodes over typed tensor values.
type Graph struct {
	Name   string
	Values []Value
	Nodes  []Node
}

// New creates an empty named graph.
func New(name string) *Graph {
	return &Graph{Name: name}
}

// AddValue appends a tensor descriptor and returns its ID.
func (g *Graph) AddValue(name string, shape Shape, dtype DType, persistent bool) ValueID {
	id := ValueID(len(g.Values))
	g.Values = append(g.Values, Value{
		ID:         id,
		Name:       name,
		Shape:      shape,
		DType:      dtype,
		Persistent: persistent,
	})
	return id
}

// AddNode appends a compute node and returns its index.
func (g *Graph) AddNode(name string, op OpKind, inputs, outputs []ValueID, attrs map[string]any) int {
	id := len(g.Nodes)
	g.Nodes = append(g.Nodes, Node{
		ID:      id,
		Name:    name,
		Op:      op,
		Inputs:  inputs,
		Outputs: outputs,
		Attrs:   attrs,
	})
	return id
}

// Value returns a pointer to the value descriptor for id.
func (g *Graph) Value(id ValueID) *Value {
	return &g.Values[id]
}

// Validate checks structural invariants: non-empty names, in-range IDs, and
// that every input value is defined (by a prior node or as persistent) before use.
func (g *Graph) Validate() error {
	nv := len(g.Values)
	defined := make([]bool, nv)
	for i, v := range g.Values {
		if v.Persistent {
			defined[i] = true
		}
	}
	for _, n := range g.Nodes {
		if n.Name == "" {
			return fmt.Errorf("node %d has empty name", n.ID)
		}
		for _, inp := range n.Inputs {
			if int(inp) >= nv {
				return fmt.Errorf("node %q: input %d out of range", n.Name, inp)
			}
			if !defined[inp] {
				return fmt.Errorf("node %q: input %q used before definition", n.Name, g.Values[inp].Name)
			}
		}
		for _, out := range n.Outputs {
			if int(out) >= nv {
				return fmt.Errorf("node %q: output %d out of range", n.Name, out)
			}
			defined[out] = true
		}
	}
	return nil
}
