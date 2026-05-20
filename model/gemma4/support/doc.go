// Package model contains Gemma4 diagnostic tests.
//
// Files in this directory are intentionally guarded by the `diagnostic` build
// tag. They are not part of the default model test/build surface, but they
// should compile under `go test -tags diagnostic ./model/gemma4` when local
// diagnostic assets are available.
//
// Backend-owned quantized helpers are imported directly from their owning
// packages (for example, backends/mlx) rather than through runtime/quant.
package model
