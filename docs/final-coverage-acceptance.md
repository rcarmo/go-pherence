# Final backend coverage acceptance tracker

This tracker maps the remaining acceptance criteria for practical backend coverage. It should be updated after the next phase-level validation run.

## Acceptance criteria

| Criterion | Status | Notes |
|---|---|---|
| All kernel coverage rows implemented or explicitly N/A with rationale | In progress | Current tables document ownership and several N/A/future paths; final pass still needed after Vulkan and hardware smoke decisions. |
| No backend-owned implementation remains in `model` or `runtime/quant` except compatibility wrappers | Mostly complete | `runtime/quant` is wrapper-only; model code imports backend-owned quant packages directly. Keep import-boundary tests active. |
| `go test ./...` passes | Pending | Deferred until full phase validation. Use `GOTMPDIR=$PWD/.gotmp`. |
| `go vet ./...` passes | Previously passed | Re-run during final validation gate. |
| CPU-only test gate passes without GPU present | Previously passed | Re-run `make test-cpu` during final validation gate. |
| NVIDIA smoke tests pass on available NVIDIA hardware | Pending hardware run | NVIDIA Q4 asymmetric and NVFP4 native packed paths are documented boundaries unless new model paths require them. |
| Vulkan smoke tests pass or are explicitly opt-in skipped | Pending | Vulkan wrappers are validating stubs; numeric parity waits for pipeline-cache wiring. CPU/software Vulkan remains opt-in. |
| Documentation reflects final package layout and kernel coverage | In progress | Docs index, layout, coverage, parity, malformed-input, validation, Vulkan, NVIDIA, BF16, NVFP4, and benchmark queue docs are current as of Phase 11. |

## Deferred implementation work

These items are intentionally not blockers for documenting current coverage, but remain open implementation tracks:

- AVX2/NEON RoPE, activation, Q4, MLX, and NVFP4 optimized kernels.
- Vulkan SPIR-V pipeline-cache wiring and availability-gated CPU-vs-Vulkan numeric parity tests.
- Native NVIDIA NVFP4 tensor-core path behind capability checks.
- NVIDIA hardware smoke refresh for BF16 LM-head, Q4 symmetric paths, MLX paths, and NVFP4 dense/native boundaries.
- Benchmark snapshot refresh for Q4, MLX, and NVFP4 additions listed in `benchmark-snapshot-queue.md`.

## Final validation sequence

When the current phase is ready to close, run the standard gate from `validation-gates.md`:

```bash
GOTMPDIR=$PWD/.gotmp go test ./...
GOTMPDIR=$PWD/.gotmp go vet ./...
make test-cpu
```

Then refresh benchmark snapshots if the test gate passes:

```bash
GOTMPDIR=$PWD/.gotmp go test ./model -run '^$' -bench 'BenchmarkCPUHot' -benchmem
```
