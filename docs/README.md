# go-pherence documentation index

This directory contains the durable backend, model, validation, and research notes for go-pherence. The top-level `README.md` stays high-level; detailed backend coverage and validation state lives here.

## Backend ownership and coverage

- [architecture.md](architecture.md) — architecture overview and package-boundary rationale.
- [backend-layout.md](backend-layout.md) — current backend/model ownership and package layout.
- [kernel-coverage.md](kernel-coverage.md) — kernel and quantization coverage across backends.
- [backend-parity-matrix.md](backend-parity-matrix.md) — scalar/reference parity targets and hardware-gated test policy.
- [malformed-input-coverage.md](malformed-input-coverage.md) — exported wrapper malformed-input coverage tracker.
- [quant-import-audit.md](quant-import-audit.md) — `runtime/quant` compatibility boundary audit.

## Validation and performance

- [validation-gates.md](validation-gates.md) — phase-level test, vet, CPU-only, and hardware smoke gates.
- [final-coverage-acceptance.md](final-coverage-acceptance.md) — final backend coverage acceptance tracker.
- [performance.md](performance.md) — benchmark notes and current snapshots.
- [benchmark-snapshot-queue.md](benchmark-snapshot-queue.md) — hot-path benchmark entrypoints and pending snapshot refreshes.
- [cpu-simd-coverage.md](cpu-simd-coverage.md) — CPU/SIMD coverage and benchmark context.

## GPU and quantized runtimes

- [gpu-options.md](gpu-options.md) — GPU compute paths and backend overview.
- [backend-selection.md](backend-selection.md) — backend selection order, gates, Vulkan wrapper status, and fallback rules.
- [vulkan-dispatch-inventory.md](vulkan-dispatch-inventory.md) — Vulkan shader/wrapper inventory.
- [vulkan-validation-plan.md](vulkan-validation-plan.md) — Vulkan pipeline-cache and parity-test plan.
- [nvidia-quant-boundaries.md](nvidia-quant-boundaries.md) — NVIDIA Q4/NVFP4 support boundaries.
- [bf16-parity.md](bf16-parity.md) — BF16 CPU/NVIDIA parity expectations.
- [nvfp4.md](nvfp4.md) — NVFP4/FP4 support track and checkpoint notes.

## Model/research notes

- [gemma4-precision.md](gemma4-precision.md) — Gemma4 GPU correctness and precision notes.
- [mtp-speculative.md](mtp-speculative.md) — Gemma4/Qwen3.6 MTP research and implementation status.
- [qwen36-mtp.md](qwen36-mtp.md) — Qwen3.6 native MTP checkpoint findings.
- [qwen35-reference-audit.md](qwen35-reference-audit.md) — Qwen3.5 reference audit.
- [orthrus.md](orthrus.md) — Orthrus/speculative decoding notes.
- [turboquant.md](turboquant.md) — TurboQuant notes.
- [weight-budget.md](weight-budget.md) — tiered weight budget manager.

## Project history

- [development-log.md](development-log.md) — development log.
- [refactor-plan.md](refactor-plan.md) — source-tree refactor status and remaining cleanup plan.
- [simd-folder-reorg.md](simd-folder-reorg.md) — SIMD folder reorganization notes.
