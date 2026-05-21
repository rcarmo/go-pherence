# Model package refactor plan

The `model` package currently owns both generic decoder machinery and architecture-specific rules for LLaMA-family, Gemma3/4, Qwen, MoE, MTP, and GPU fallback paths. Future refactors should avoid mechanical subpackage moves that break unexported package-local contracts; instead, promote stable common contracts first and then move architecture-owned loaders/forward paths behind explicit bridge APIs.

## Target layout

```text
model/
  common/      # architecture-neutral config/types/check helpers
  llama/       # LLaMA-family architecture-specific helpers
  gemma/       # Gemma3/Gemma4 config/load/forward rules
  qwen/        # Qwen-specific loaders/native MTP helpers
  speculative/ # generic speculative/MTP orchestration contracts
  gpu/         # shared GPU residency/fallback orchestration
```

The existing root `model` package should remain a compatibility facade until CLI and downstream imports are migrated.

## Migration rules

1. Move pure metadata/types first, with root-package aliases where possible.
2. Do not move files that depend on unexported helpers until those helpers have explicit exported bridge APIs.
3. Keep architecture-specific config normalization out of the generic loader once a dedicated package exists.
4. Keep tests close to the owning package and run `GOTMPDIR=$PWD/.gotmp go test ./... -run '^$'` after each slice.
5. Do not repeat broad mechanical subpackage splits; they were reverted because assembly/package-local symbols and unexported model contracts are not yet bridged.

## First completed slice

`model/common` now owns the generic `Config` metadata type and its architecture-neutral helper methods. The root `model.LlamaConfig` is a type alias for compatibility.

This enables the next safe slice: moving Gemma4 nested config normalization and per-layer KV-head/K=V rules behind a dedicated Gemma loader/config package while keeping root `LoadLlama` as a compatibility entrypoint.
