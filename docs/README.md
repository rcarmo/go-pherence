# go-pherence documentation

The codebase covers ordinary local inference, large-model placement, speech, image generation and several experimental accelerators. This index starts with things you can run, then moves down through architecture, model-specific boundaries, validation and the historical notes that explain some of the stranger corners.

## Start with a task

| I want to... | Read |
|---|---|
| Run a command or service | [Commands](commands.md) |
| Check whether a model or format is supported | [Supported models](supported-models.md) |
| Choose CPU, NVIDIA or another backend | [Backend selection](backend-selection.md) and [Tuning](tuning.md) |
| Understand package ownership and execution flow | [Architecture](architecture.md), [Backend stack](backend-stack.md) and [Backend layout](backend-layout.md) |
| Design or optimise an LLM inference runtime | [Practical LLM inference blueprint](llm-inference-blueprint.md) |
| Transcribe or translate audio | [Whisper and translated VTT](whisper-diarize-vtt.md) |
| Transcribe with speaker labels and timestamps | [MOSS transcription and diarisation](moss-transcribe-diarize.md) |
| Work on speculative decoding or MTP | [MTP and speculative decoding](mtp-speculative.md) |
| Run the standard correctness gates | [Validation gates](validation-gates.md) |
| Compare performance | [Performance](performance.md) and [matmul optimisation results](matmul-optimisation-results.md) |

## Architecture and backends

[Architecture](architecture.md) is the canonical high-level view. It explains the checked scalar/SIMD baseline, model layer and backend boundary; the generated SVG near the top is the visual version of that path.

[Backend stack](backend-stack.md) describes what each backend owns. The [practical LLM inference blueprint](llm-inference-blueprint.md) generalises the request-state, scheduling, caching, batching, speculation and KV promotion sequence used for new inference work. The narrower reference pages cover [source layout](backend-layout.md), [runtime selection and fallback](backend-selection.md), [kernel coverage](kernel-coverage.md), [GPU options](gpu-options.md) and [weight placement](weight-budget.md).

Quantised and hardware-specific references:

* [TurboQuant](turboquant.md) covers compressed KV and scratch policy.
* [NVIDIA quantisation boundaries](nvidia-quant-boundaries.md), [NVFP4](nvfp4.md) and [BF16 parity](bf16-parity.md) state the numerical contracts for those formats.
* [SpacemiT IME2](spacemit-ime2.md) covers the K3/CIX accelerator path.
* [Vulkan inventory](vulkan-dispatch-inventory.md) and [Vulkan validation](vulkan-validation-plan.md) describe the current experimental boundary.
* [Quant import audit](quant-import-audit.md) documents the compatibility facade around quantised backends.

## Model and feature guides

### Language models and speculative decoding

* [Gemma4 31B runbook](gemma4-31b-runbook.md) -- local E4B/31B placement and smoke strategy.
* [MTP and speculative decoding](mtp-speculative.md) -- canonical implementation and validation page.
* [Qwen3.6 MTP](qwen36-mtp.md) -- checkpoint-specific native-MTP work.
* [Qwen3-TTS](qwen3-tts-support.md) and [LFM2 MoE](lfm2-moe-support.md) -- current loader/runtime boundaries.

### Speech

* [Whisper and translated VTT](whisper-diarize-vtt.md) -- user-facing pipeline, media handling and resume behaviour.
* [Whisper model assets](whisper-model-assets.md) -- exact checkpoints and tensor shapes.
* [Whisper execution graph](whisper-execution-graph.md) -- backend coverage and parity details.
* [MOSS transcription and diarisation](moss-transcribe-diarize.md) -- pinned native graph, output formats and measured CPU/GPU performance.

### Vision, image and 3D

* [DiffusionGemma](diffusiongemma-support.md) -- canonical support page for block-diffusion text generation.
* [Ideogram 4](ideogram4-support.md) and [Ideogram on SpacemiT](ideogram4-spacemit.md).
* [MiniCPM-V/O](minicpmv-support.md) and its [runtime roadmap](minicpmv-runtime-roadmap.md).
* [Hunyuan3D-2](hunyuan3d-2-support.md), [Trellis2](trellis2-support.md) and [Z-Image-Turbo](zimage-turbo-support.md).

[Model coverage status](model-coverage-status.md) is the compact engineering tracker. The generated [coverage snapshot](model-coverage-snapshot.md) is useful for tooling and review, but [Supported models](supported-models.md) is the reader-facing answer.

## Validation and performance

[Validation gates](validation-gates.md) is the canonical command list. [Validation hardening](validation-hardening.md), [malformed-input coverage](malformed-input-coverage.md) and the [backend parity matrix](backend-parity-matrix.md) explain why those checks exist and which failures are hardware- or asset-dependent.

Performance references:

* [Performance](performance.md) -- current model and backend measurements.
* [SIMD matmul policy](simd-matmul.md) -- shape-aware decode/prefill dispatch.
* [Matmul audit](matmul-audit.md), [benchmark protocol](matmul-benchmark-protocol.md) and [final results](matmul-optimisation-results.md) -- the complete cache/register-tiling programme and its retained or rejected outcomes.
* [TurboFieldfare audit](turbo-fieldfare-audit.md) and [adoption results](turbo-fieldfare-adoption-results.md) -- transferable expert-streaming, KV, attention and prefill techniques, plus the measured go-pherence adoption baseline.
* [Whisper on RISC-V](whisper-riscv-optimization.md) -- measured K3/RVV/IME path.

The generated [test matrix](test-matrix.svg) is a scoped visual guide, not a count of the entire repository. Regenerate both checked-in diagrams with `make docs-diagrams`.

## History and diagnostics

These pages preserve investigations, generated snapshots and implementation plans. They are valuable when debugging a numerical boundary, but they should not be used as current feature summaries.

### Project history and refactors

* [Development log](development-log.md)
* [Refactor plan](refactor-plan.md)
* [SIMD folder reorganisation](simd-folder-reorg.md)
* [Model package refactor](model-package-refactor.md)
* [Reusable component consolidation](reusable-component-consolidation.md)
* [Final coverage acceptance](final-coverage-acceptance.md)
* [Benchmark snapshot queue](benchmark-snapshot-queue.md)
* [CPU/SIMD coverage snapshot](cpu-simd-coverage.md)

### Model investigations

* [DiffusionGemma status snapshot](diffusiongemma-status.md), [llama.cpp alignment](diffusiongemma-llamacpp-alignment.md) and [GGUF GPU profile](diffusiongemma-gguf-gpu-profile.md)
* [Gemma llama.cpp audit](gemma-llamacpp-audit.md), [Gemma4 alignment](gemma4-llamacpp-alignment.md), [MTP benchmarks](gemma4-mtp-benchmarks.md), [first divergence](gemma4-mtp-first-diff.md) and [precision notes](gemma4-precision.md)
* [Qwen3.5 reference audit](qwen35-reference-audit.md), [Orthrus notes](orthrus.md) and [KVBoost application plan](kvboost-application-plan.md)
* [Original Whisper plan](whisper-plan.md)
* [IME2 I8/I4 port notes](ime2-i8i4-port-notes.md)

## Diagrams

![Core LLM and NVIDIA execution path](architecture.svg)

![Focused backend validation matrix](test-matrix.svg)

The SVGs are generated from `scripts/render-architecture.ts` and `scripts/render-test-matrix.ts`. Keep labels and status claims in the scripts; editing the rendered SVGs by hand only guarantees they will drift again.
