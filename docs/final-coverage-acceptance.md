# Final backend coverage acceptance tracker

This tracker maps the remaining acceptance criteria for practical backend coverage. It should be updated after the next phase-level validation run.

## Acceptance criteria

| Criterion | Status | Notes |
|---|---|---|
| All kernel coverage rows implemented or explicitly N/A with rationale | In progress | Current tables document ownership and several N/A/future paths; final pass still needed after Vulkan and hardware smoke decisions. |
| No backend-owned implementation remains in `model` or `runtime/quant` except compatibility wrappers | Mostly complete | `runtime/quant` is wrapper-only; model code imports backend-owned quant packages directly. Keep import-boundary tests active. |
| `go test ./...` passes | Passed | Full gate passed earlier on 2026-05-20; compile-only sweep `GOTMPDIR=$PWD/.gotmp go test ./... -run '^$'` also passed after the SIMD/tensor/BERT boundary cleanup. |
| `go vet ./...` passes | Passed | `GOTMPDIR=$PWD/.gotmp go vet ./...` passed on 2026-05-20 and was refreshed cleanly after the SIMD/tensor/BERT checked-boundary cleanup. |
| CPU-only test gate passes without GPU present | Passed | `make test-cpu` passed on 2026-05-20. |
| NVIDIA smoke tests pass on available NVIDIA hardware | Pending hardware run | NVIDIA Q4 asymmetric and NVFP4 native packed paths are documented boundaries unless new model paths require them. |
| Vulkan smoke tests pass or are explicitly opt-in skipped | Pending hardware/runtime run | Vulkan wrappers have validating dispatch paths and availability-gated parity coverage; CPU/software Vulkan remains opt-in. |
| Documentation reflects final package layout and kernel coverage | Current, pending validation fallout | Docs index, layout, coverage, parity, malformed-input, validation, Vulkan, NVIDIA, BF16, NVFP4, benchmark queue, and development log are current as of Phase 12; revisit after the phase validation gate. |

## Deferred implementation work

These items are intentionally not blockers for documenting current coverage, but remain open implementation tracks:

- AVX2/NEON RoPE, activation, Q4, MLX, and NVFP4 optimized kernels.
- Vulkan hardware/runtime smoke refresh for the existing SPIR-V pipeline-cache wiring and availability-gated parity tests.
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
