# Architecture

![Core go-pherence inference path](architecture.svg)

go-pherence keeps model code separate from the machinery that loads weights and runs kernels. The split is intentionally plain: commands prepare input and choose policy, model packages own graph semantics, loaders own checkpoint formats, and backends own arithmetic. Every accelerated boundary has a checked CPU implementation or a model-level parity fixture.

## The path through the code

A normal inference request moves through four layers:

1. A command under `cmd/` parses user input and chooses model, placement and generation options.
2. `model/` and `models/` implement decoder, encoder, speech and model-specific graph semantics.
3. `loader/` reads configuration, tokenizers and weights from safetensors, GGUF or model sidecars.
4. `tensor`, `runtime` and `backends` execute the graph on CPU, NVIDIA, Vulkan or embedded hardware.

This is a dependency boundary rather than a promise that every model uses the same graph engine. LLaMA-family generation is deliberately direct, Whisper has its own encoder/decoder structure, and image models use model-specific schedulers. The shared pieces are loaders, checked kernels, memory policy and validation conventions.

## Package ownership

| Area | Owner | What belongs there |
|---|---|---|
| Commands | `cmd/llm`, `cmd/qwen`, `cmd/audio`, `cmd/image`, `cmd/models`, `cmd/spacemit` | Flags, files, prompts, network I/O, board tools and user-facing reporting |
| Model semantics | `model`, `models/*` | Layers, attention, generation loops, schedulers and model-specific state |
| Configuration and weights | `loader/config`, `loader/tokenizer`, `loader/weights`, `loader/safetensors`, `loader/gguf` | Checkpoint metadata, tokenization, mmap/shards and quantised layouts |
| Tensor/runtime support | `tensor`, `runtime/kv`, `runtime/memory` | Lazy tensor operations, KV state, compression, residency and rollback |
| CPU execution | `backends/simd/runtime`, `backends/simd/kernels`, `backends/simd/quant/*` | Checked dispatch, scalar references, AVX2/NEON/RVV assembly and quantised CPU kernels |
| NVIDIA execution | `backends/nvidia/runtime`, `backends/nvidia/ptx`, `backends/cuda/ptx`, `backends/nvidia/ioctl` | Driver loading, buffers/streams, runtime dispatch, shared/Whisper PTX assets and direct-driver experiments |
| Other accelerators | `backends/vulkan`, `backends/spacemit` | SPIR-V experiments, RVV, IME2 and CIX/SpacemiT helpers |
| Placement | `backends/placement` | Backend-neutral memory budgets and layer-placement estimates |

[Backend layout](backend-layout.md) has the detailed source-tree map. [Backend stack](backend-stack.md) describes the execution backends and their current readiness.

## Backend selection

CPU execution is the correctness baseline. SIMD wrappers validate dimensions and backing storage before entering assembly, then fall back to scalar Go when the ISA or shape is unsupported.

NVIDIA support loads `libcuda.so.1` at runtime through purego. The binary embeds PTX from `backends/nvidia/ptx` and the Whisper-oriented `backends/cuda/ptx`, then asks the driver to compile it for the installed GPU, and keeps model state resident where the model-specific path has passed parity. A failed initialisation or operation returns to CPU unless the command documents a stricter requirement.

Vulkan and SpacemiT are explicit secondary paths. Vulkan has working buffer/shader dispatch but is not a general model backend; several parity gates remain open. SpacemiT combines ordinary RVV CPU kernels with IME2/AICPU experiments and therefore has its own hardware-specific validation.

The exact order and command switches are in [Backend selection](backend-selection.md). Memory placement and cache controls are in [Tuning](tuning.md) and [Weight budgets](weight-budget.md).

## Weight formats

The model loader normalises metadata but does not eagerly convert every checkpoint to F32. Keeping the original layout avoids needless memory traffic and lets each backend choose an appropriate kernel.

```text
checkpoint
  |
  +-- MLX affine 4-bit -> backends/mlx
  +-- GPTQ 4-bit       -> backends/simd/quant/q4 or NVIDIA Q4
  +-- GGUF blocks      -> loader/gguf + backend-owned row/batch kernels
  +-- BF16/F16/F32     -> tensor or model-owned dense weights
  +-- FP8/NVFP4        -> model-specific CPU/NVIDIA paths
```

`runtime/quant` is a compatibility facade; new backend code should import the owning package directly. [Supported models](supported-models.md) maps formats to model families, while [NVIDIA quantisation boundaries](nvidia-quant-boundaries.md), [NVFP4](nvfp4.md) and [BF16 parity](bf16-parity.md) cover the numerical details.

## Correctness before dispatch

Unsafe boundaries validate dimensions, strides, byte counts and pointer state before assembly or driver calls. Scalar references own the contract for individual kernels; real-model fixtures own boundaries where different reduction orders make bitwise equality unrealistic.

That policy matters because a fast kernel is easy to benchmark and surprisingly easy to enable too early. The repository therefore keeps rejected candidates alongside explicit parity gates when they are useful for further work, but default dispatch only uses paths that passed the relevant fixture.

The canonical checks live in [Validation gates](validation-gates.md). [Validation hardening](validation-hardening.md) explains the shared guard policy, and [Backend parity matrix](backend-parity-matrix.md) identifies the reference implementation for each backend surface.

## The parts that are deliberately model-specific

Speculative decoding, MTP, MoE expert scheduling, speech timestamps and diffusion schedulers do not belong in a generic tensor abstraction merely because they contain matrix multiplication. They stay close to their model packages and use shared backend primitives underneath.

Current model boundaries and limitations are documented in [Supported models](supported-models.md). Deep implementation investigations -- including older refactor notes and numerical first-divergence reports -- are indexed under "History and diagnostics" in the [documentation landing page](README.md).
