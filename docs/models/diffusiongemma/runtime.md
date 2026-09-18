# DiffusionGemma runtime and backends

[Support guide](README.md) | [Vision limits](vision.md) | [Validation](validation.md)

## Execution paths

| Path | What it uses | Flags | Current limit |
| --- | --- | --- | --- |
| Safetensor CPU sparse text | Full HF text shards with the CPU/SIMD dispatcher | `-model DIR -cpu-dispatcher -allow-slow-cpu -lm-head-top-k 8` | Requires all 11 shards. The sparse top-k option is an experimental/debug path, not proof of full-logit parity. |
| GGUF text runtime | llama.cpp-compatible `Q4_K_M` GGUF weights, plus metadata/tokenizer from `-model` | `-model DIR -gguf-model FILE -cpu-dispatcher` or `-gpu-dispatcher` | Text is implemented today, but this still does not mean full multimodal parity. |
| NVIDIA / CUDA GPU | `GPUDispatcher` offloads GEMV projections to CUDA; attention, norms, and sampling remain CPU-owned unless a narrower device-resident path is enabled | `-model DIR -gpu-dispatcher` | GPU prompt prefill requires CUDA SGEMM. If SGEMM is unavailable, the dispatcher falls back to CPU unless a strict GGUF backend-graph path has been requested. |
| FP8 CUDA text | GPU dispatcher plus an FP8-dynamic checkpoint directory | `-model DIR -gpu-dispatcher -fp8-model DIR` | NVIDIA-only. `-fp8-expert-prewarm-layers` defaults to `9`; expert VRAM still needs budgeting. |
| K3 native / A100 Q80 | riscv64 K3 RVV/X100 primitives, with optional A100 row-scale `Q80x32` projection packing | `-model DIR -cpu-dispatcher -allow-slow-cpu -k3` and optionally `-k3-a100-q8` | riscv64 only. `-k3-a100-workers` is capped at `8`, `-k3-q80-prewarm-experts` is memory-heavy, and the K3 path is about text execution rather than multimodal parity. |

## CLI examples

Use `-denoise-steps` to change the effective denoising loop. `-diffusion-steps` is recorded separately for reporting and does not replace it.

```bash
go run ./cmd/diffusiongemmainspect \
  -model /path/to/diffusiongemma \
  -require-text-sparse-ready
```

```bash
go run ./cmd/diffusiongemmarun \
  -model /path/to/diffusiongemma \
  -cpu-dispatcher -allow-slow-cpu \
  -prompt "hi" \
  -max-new 1 -canvas 1 \
  -denoise-steps 1 \
  -residency-budget-gib 16 \
  -lm-head-top-k 8 \
  -decode
```

```bash
go run ./cmd/diffusiongemmarun \
  -model /path/to/diffusiongemma-metadata \
  -gguf-model /path/to/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -gpu-dispatcher \
  -prompt "hi" \
  -max-new 1 -canvas 1 \
  -denoise-steps 1 \
  -resident-layers 1 \
  -decode
```

```bash
go run ./cmd/diffusiongemmarun \
  -model /path/to/diffusiongemma \
  -gpu-dispatcher \
  -fp8-model /path/to/diffusiongemma-fp8 \
  -prompt "hi" \
  -max-new 1 -canvas 1 \
  -denoise-steps 1 \
  -fp8-expert-prewarm-layers 9 \
  -decode
```

```bash
go run ./cmd/diffusiongemmarun \
  -model /path/to/diffusiongemma \
  -cpu-dispatcher -allow-slow-cpu \
  -k3 -k3-a100-q8 \
  -k3-threads 8 \
  -k3-a100-workers 6 \
  -k3-q80-residency-budget-gib 8 \
  -prompt "hi" \
  -max-new 1 -canvas 1 \
  -denoise-steps 1 \
  -decode
```

When `-k3` and `-k3-a100-q8` are both set, `cmd/diffusiongemmarun` auto-enables the K3 LM-head preset unless you override it yourself: `-lm-head-top-k` defaults to `64`, and `-k3-a100-lmhead` plus `-k3-a100-lmhead-prefetch` are switched on for you.


These are command forms, not fresh successful inference runs. The [historical logs](../../history/diffusiongemma/README.md) identify the assets and revisions used for earlier measurements. Do not treat their old sparse outputs as parity for the current runtime.
