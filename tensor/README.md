# tensor

A lazy, UOP-based tensor compute graph: tensors are built as a graph of micro-ops
(`uop.go`) that is rewritten/fused and then realized to concrete buffers.

| Area | Files |
|---|---|
| Graph core | `tensor.go`, `uop.go`, `ops.go`, `dtype.go`, `shape.go`, `unsafe.go` |
| Rewrite / fusion | `rewrite.go`, `rules.go`, `pattern.go`, `fuse.go`, `realize.go` |
| NN ops | `nn.go`, `modules.go`, `matmul.go`, `conv1d.go`, `embedding.go`, `pool.go`, `broadcast.go` |
| Safety | `checked.go` |

Low-level reusable compute primitives live in `backends/simd` and
`backends/spacemit`; `tensor` is the architecture-independent graph/dispatch layer
above them.
