# Final backend coverage acceptance tracker

This tracker maps the remaining acceptance criteria for practical backend coverage. Last reviewed on 2026-05-21 after the tree-split recovery, guard sweep, documentation refresh, Gemma4 packed-MTP smoke validation, E4B MTP target validation, and README documentation split.

## Acceptance criteria

| Criterion | Status | Notes |
|---|---|---|
| All kernel coverage rows implemented or explicitly N/A with rationale | Locally current | Current tables document owner packages, checked runtime APIs, compatibility boundaries, malformed-input/overflow guards, and hardware/future paths with rationale. Remaining changes are hardware smoke or new optimized-kernel work. |
| No backend-owned implementation remains in `model` or `runtime/quant` except compatibility wrappers | Locally complete | `runtime/quant` is wrapper-only; model/tensor/BERT code imports backend-owned packages and checked SIMD runtime APIs directly. Import-boundary tests pass, and `go test -tags diagnostic ./model/gemma4 -run '^$'` compiles cleanly. |
| `go test ./...` passes | Passed | `GOTMPDIR=$PWD/.gotmp go test ./...` passed on 2026-05-21 after the Gemma4 31B packed-MTP smoke wiring. |
| `go vet ./...` passes | Passed | `GOTMPDIR=$PWD/.gotmp go vet ./...` passed on 2026-05-21 after the Gemma4 31B packed-MTP smoke wiring. |
| CPU-only test gate passes without GPU present | Passed | `make test-cpu` passed after the Qwen MTP, MoE, KV, and NVIDIA runtime guard sweep. |
| NVIDIA smoke tests pass on available NVIDIA hardware | Pending hardware run; local package gate clean | `GOTMPDIR=$PWD/.gotmp go test ./backends/nvidia/...` passes locally with availability-gated tests. Hardware smoke on a CUDA-capable host remains the only open NVIDIA acceptance item. |
| Vulkan smoke tests pass or are explicitly opt-in skipped | Pending hardware/runtime run; local package gate clean | `GOTMPDIR=$PWD/.gotmp go test ./backends/vulkan` passes locally with validating wrappers and availability-gated parity. CPU/software Vulkan remains opt-in; real-device smoke remains hardware/runtime-dependent. |
| Documentation reflects final package layout and kernel coverage | Current | Docs index, layout, backend stack, supported models, commands, validation hardening, coverage, parity, malformed-input, Vulkan, NVIDIA, BF16, NVFP4, benchmark queue, performance snapshots, Gemma4 MTP runbooks, and Qwen MTP candidate notes are current after refreshed gates plus Qwen native-MTP, speculative decode sizing, MoE expert-upload, KV compressed sizing, DevBuf, PTX loader, mega-module guard updates, E4B long-prompt smoke, and the 31B packed-MTP smoke. |

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
