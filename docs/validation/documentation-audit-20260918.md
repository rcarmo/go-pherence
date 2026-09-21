# Documentation and host-check audit, 2026-09-18

Historical validation snapshot. Preserve the recorded results below; later fixes and current limitations are in the [repository safety audit](repository-safety-audit-20260919.md). Earlier failed host checks were subsequently corrected. Any GPU success here predates the later bus-loss incidents and is not validation of the current candidate.

Current guides are organised under `guides/`, `models/`, `speech/`, `backends/`, `architecture/`, `validation/` and `performance/`. Seventy-seven existing pages were moved into those folders or `history/`; the command catalogue was split by task and DiffusionGemma now has separate runtime, vision and validation guides. The previous DiffusionGemma support diary is retained in [history](../history/diffusiongemma/implementation-log.md).

Model summaries distinguish native Jevlike/GLiNER support from upstream API compatibility, and DiffusionGemma text execution from complete multimodal readiness. Ideogram's CPU reference path and opt-in partial NVIDIA offload are described separately. Older performance numbers remain workload-specific historical observations. No new broad performance claim or full DiffusionGemma parity claim is made.

The coverage generator, its tests and manifest use the actual Qwen3-TTS/LFM2 command locations. Generated coverage still covers a limited set of families, not the whole repository. Benchmark JSON evidence, upstream provenance and the original Gemma4 audit artefacts were left unchanged. Markdown links were adjusted where documents moved.

## Verification

| Check | Result |
|---|---|
| `make docs-check` | Link-checker regressions, local links, generated diagrams and manifest tests pass; new Markdown and reference-style links are included |
| `make model-coverage-snapshot-check` | Generated snapshot matches |
| Command-path scan | 124 current-document `go run ./cmd/...` references resolve; historical commands excluded |
| Focused Go test/vet | Coverage generator, docs, MiniCPM roadmap reporting, Jevlike and GLiNER packages/CLIs pass |
| CLI help | DiffusionGemma runner/inspector, GLiNER, Jevlike synthetic/train/eval pass |
| CLI examples | Jevlike 64/16/16 synthetic split, three-epoch training and evaluation pass; published GLiNER checkpoint extracts Alice, Acme and Lisbon |
| `go build ./...` | Passes on Linux/amd64, also with CGo disabled |
| `go test ./backends/...` | Passes on this host |
| `make spacemit-host-check` | Build, vet and host tests pass |
| `make spacemit-cross-compile` | Linux/RISC-V packages and selected test binaries compile; none executed |
| Linux/ARM64 cross-build | SpacemiT packages/commands and Jevlike/GLiNER CLIs compile; none executed |
| `make spacemit-hardware-test` on amd64 | Rejects the host before executing hardware tests, as intended |
| `git diff --check` | Passes |

The Jevlike smoke demonstrates CLI correctness, not useful model accuracy after three small epochs. Existing [Jevlike](../../model/jevlike/VALIDATION.md) and [GLiNER](../../model/gliner2/VALIDATION.md) parity/benchmark records retain their own workload scope.

## Remaining failures

`make host-test` still fails in `model` (GGUF missing-router and MTP fixtures) and `model/qwen` (Qwen3.5 tensor/shape fixtures). Whole-tree vet reports `backends/nvidia/runtime/runtime.go:546` unsafe-pointer use and `model/diffusiongemma/q6_i8dot_amd64.s:29` using `VMOVD` for an int32 return. No full-suite success is claimed.

The SpacemiT fix uses platform constraints, unavailable host stubs and explicit K3 hardware-test opt-in. It does not drop whole backend directories from validation. Portable IME tests remain enabled and exposed an existing scalar packed-GEMM layout mismatch; matching the interleaved 4x8 layout fixes those regressions without changing native assembly.

[Validation gates](validation-gates.md) | [Documentation index](../README.md)
