# go-pherence documentation index

This directory contains the durable backend, model, validation, and research notes for go-pherence. The top-level `README.md` stays high-level; detailed backend coverage and validation state lives here.

## Backend ownership and coverage

- [architecture.md](architecture.md) — architecture overview and package-boundary rationale.
- [backend-stack.md](backend-stack.md) — NVIDIA, Vulkan, SIMD, BF16, and package ownership summary.
- [backend-layout.md](backend-layout.md) — current backend/model ownership and package layout.
- [kernel-coverage.md](kernel-coverage.md) — kernel and quantization coverage across backends.
- [backend-parity-matrix.md](backend-parity-matrix.md) — scalar/reference parity targets and hardware-gated test policy.
- [malformed-input-coverage.md](malformed-input-coverage.md) — exported wrapper malformed-input coverage tracker.
- [validation-hardening.md](validation-hardening.md) — readable malformed-input and boundary-hardening summary.
- [quant-import-audit.md](quant-import-audit.md) — `runtime/quant` compatibility boundary audit.

## Validation and performance

- [validation-gates.md](validation-gates.md) — phase-level test, vet, CPU-only, and hardware smoke gates.
- [commands.md](commands.md) — CLI usage, MTP smokes, Qwen MTP triage, and benchmark harnesses.
- [final-coverage-acceptance.md](final-coverage-acceptance.md) — final backend coverage acceptance tracker.
- [performance.md](performance.md) — benchmark notes and current snapshots.
- [benchmark-snapshot-queue.md](benchmark-snapshot-queue.md) — hot-path benchmark entrypoints and refreshed snapshot status.
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

- [supported-models.md](supported-models.md) — supported architectures, formats, and performance snapshot.
- [hunyuan3d-2-support.md](hunyuan3d-2-support.md) — Hunyuan3D-2 feasibility assessment and staged implementation plan.
- [gemma4-precision.md](gemma4-precision.md) — Gemma4 GPU correctness and precision notes.
- [gemma4-31b-runbook.md](gemma4-31b-runbook.md) — local Gemma4 31B main/MTP assets and current run strategy.
- [model-package-refactor.md](model-package-refactor.md) — safe plan for splitting generic model contracts from architecture-specific packages.
- [mtp-speculative.md](mtp-speculative.md) — Gemma4/Qwen3.6 MTP research and implementation status.
- [qwen36-mtp.md](qwen36-mtp.md) — Qwen3.6 native MTP checkpoint findings.
- [qwen35-reference-audit.md](qwen35-reference-audit.md) — Qwen3.5 reference audit.
- [orthrus.md](orthrus.md) — Orthrus/speculative decoding notes.
- [turboquant.md](turboquant.md) — TurboQuant notes.
- [kvboost-application-plan.md](kvboost-application-plan.md) — how KVBoost-style chunked KV reuse/page-offload maps to go-pherence.
- [weight-budget.md](weight-budget.md) — tiered weight budget manager.

## Project history

- [development-log.md](development-log.md) — development log.
- [refactor-plan.md](refactor-plan.md) — source-tree refactor status and remaining cleanup plan.
- [simd-folder-reorg.md](simd-folder-reorg.md) — SIMD folder reorganization notes.
