# Final backend coverage acceptance tracker

This tracker maps the remaining acceptance criteria for practical backend coverage. It should be updated after the next phase-level validation run.

## Acceptance criteria

| Criterion | Status | Notes |
|---|---|---|
| All kernel coverage rows implemented or explicitly N/A with rationale | In progress | Current tables document ownership and several N/A/future paths; final pass still needed after Vulkan and hardware smoke decisions. |
| No backend-owned implementation remains in `model` or `runtime/quant` except compatibility wrappers | Mostly complete | `runtime/quant` is wrapper-only; model/tensor/BERT code imports backend-owned packages and checked SIMD runtime APIs directly. Import-boundary tests pass, and `go test -tags diagnostic ./model/gemma4 -run '^$'` compiles cleanly. |
| `go test ./...` passes | Passed | `GOTMPDIR=$PWD/.gotmp go test ./...` passed again after the SIMD/tensor/BERT boundary cleanup and CPU-only gate refresh on 2026-05-20. |
| `go vet ./...` passes | Passed | `GOTMPDIR=$PWD/.gotmp go vet ./...` passed on 2026-05-20 and was refreshed cleanly after the SIMD/tensor/BERT checked-boundary cleanup. |
| CPU-only test gate passes without GPU present | Passed | `make test-cpu` passed on 2026-05-20 and was refreshed cleanly after the SIMD/tensor/BERT boundary cleanup. |
| NVIDIA smoke tests pass on available NVIDIA hardware | Pending hardware run | NVIDIA Q4 asymmetric and NVFP4 native packed paths are documented boundaries unless new model paths require them. |
| Vulkan smoke tests pass or are explicitly opt-in skipped | Pending hardware/runtime run | Vulkan wrappers have validating dispatch paths and availability-gated parity coverage; CPU/software Vulkan remains opt-in. |
| Documentation reflects final package layout and kernel coverage | Current, pending validation fallout | Docs index, layout, coverage, parity, malformed-input, validation, Vulkan, NVIDIA, BF16, NVFP4, benchmark queue, and development log are current as of Phase 12; revisit after the phase validation gate. |

## Deferred implementation work

These items are intentionally not blockers for documenting current coverage, but remain open implementation tracks:

- AVX2/NEON RoPE, activation, Q4, MLX, and NVFP4 optimized kernels.
- Vulkan hardware/runtime smoke refresh for the existing SPIR-V pipeline-cache wiring and availability-gated parity tests.
- Native NVIDIA NVFP4 tensor-core path behind capability checks.
- NVIDIA hardware smoke refresh for BF16 LM-head, Q4 symmetric paths, MLX paths, and NVFP4 dense/native boundaries.
- Future benchmark snapshot refreshes after new AVX2/NEON, NVIDIA, or Vulkan hardware paths land. The current Q4, MLX, NVFP4, RoPE, activation, RMSNorm, and GQA CPU-hot snapshots were refreshed on 2026-05-20.

## Final validation sequence

When the current phase is ready to close, run the standard gate from `validation-gates.md`:

```bash
GOTMPDIR=$PWD/.gotmp go test ./...
GOTMPDIR=$PWD/.gotmp go vet ./...
make test-cpu
```

Refresh benchmark snapshots after substantive hot-path changes:

```bash
GOTMPDIR=$PWD/.gotmp go test ./model -run '^$' -bench 'BenchmarkCPUHot' -benchmem
```

The CPU-hot matrix was refreshed on 2026-05-20 and recorded in `docs/performance.md`, `docs/cpu-simd-coverage.md`, and `docs/benchmark-snapshot-queue.md`.
