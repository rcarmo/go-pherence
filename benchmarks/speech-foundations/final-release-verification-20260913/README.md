# Final supported release verification — 13 September 2026

## Revision and environment

- Tested revision: `08213a40403951d4bd25ac2e99d8c901cc47604f` (`feat/speech-simd-vulkan`).
- Public `origin/feat/speech-simd-vulkan` resolved to the same commit after an explicit ref refresh.
- Linux/amd64, bundled Go toolchain, `CGO_ENABLED=0`, `GOMAXPROCS=2`, `GO_PHERENCE_DISABLE_NVIDIA=1`.
- GNU Make is absent in this container. `run-supported-checks.sh` executes the exact recipes from the 14 declared supported speech Make targets; it does not substitute different package or test selections.
- No persistent service or native GPU/model job was started. Previously retained physical-Iris and trained-model evidence was checksum-verified separately.

## Supported gates

All 14 gates passed. `supported-status.tsv` records target, exit status and elapsed whole seconds; `logs/` retains command traces and outputs.

- `speech-foundations-check`
- `speech-vulkan-offline-check`
- `speech-vulkan-community-check`
- `speech-vulkan-community-server-check`
- `speech-vulkan-encoder-check`
- `speech-sincnet-fma-check`
- `speech-community-gemm-check`
- `speech-job-check`
- `speech-job-http-check`
- `speech-job-cli-check`
- `speech-job-serve-check`
- `speech-media-integration`
- `speech-quality-freeze-check`
- `speech-community-corpus-contract-check`

## Linux/ARM64 compile boundary

Compile-only tests passed for:

- `./models/speaker/community1`
- `./models/whisper`
- `./loader/audio/media`
- `./runtime/speechjob`
- `./runtime/speechjob/httpapi`
- `./cmd/audio/speechjob`

`./cmd/audio/speechjobserve` remains intentionally unsupported on Linux/ARM64 because trained Community/server runtime types are currently guarded by `linux && amd64`. Its nonzero compile output is retained in `arm64/speechjobserve.log` and classified as the expected architecture boundary, not a passing gate. ARM enablement is deferred and requires architecture-specific construction/kernels, ownership validation and real-device execution.

## Evidence integrity and decision

All nine retained 13 September speech evidence manifests validated with `sha256sum -c`, including the final qualification audit, go-264 promotion, trained combined service, strict Community diagnostics and SIMD/Vulkan optimisation evidence.

This verifies the supported implementation and repository evidence. It does **not** change the complete-plan decision to qualified. Strict Community-1 intermediate/default-tie gates, the Community latency target, representative multilingual ASR, broad annotated diarization/SA-WER, comparable long-form complete-pipeline performance and continuous energy evidence remain failed, unsupported or missing. The closure result is therefore a completed audit with an explicit **not qualified** release decision, not a claim that every P0–P8 acceptance gate passes.
