# Z-Image-Turbo support plan

`Tongyi-MAI/Z-Image-Turbo` is a Diffusers text-to-image pipeline, not a
LLaMA-family decoder. Native go-pherence support must therefore be a new image
generation pipeline rather than a wrapper around `llama.cpp` or Python
Diffusers.

## Published model shape

Current Hugging Face metadata identifies:

```text
pipeline:     ZImagePipeline
transformer:  ZImageTransformer2DModel (S3-DiT)
scheduler:    FlowMatchEulerDiscreteScheduler
text encoder: Qwen3Model
tokenizer:    Qwen2Tokenizer
vae:          AutoencoderKL
```

Transformer summary from the published config:

```text
dim=3840
layers=30
refiner_layers=2
heads=30
kv_heads=30
in_channels=16
cap_feat_dim=2560
axes_dims=[32 48 48]
axes_lens=[1536 512 512]
```

Text encoder summary:

```text
model_type=qwen3
hidden_size=2560
layers=36
vocab_size=151936
```

## Native implementation requirement

Support should be implemented in pure Go with backend-owned kernels:

- scalar reference kernels first, with deterministic fixtures;
- AVX2/FMA assembly or checked SIMD facade paths on `amd64`;
- NEON assembly or checked SIMD facade paths on `arm64`;
- RVV-gated assembly/facade paths on `riscv64` where Go/RVV support is
  available;
- no Python/Diffusers runtime dependency for generation.

The existing `backends/simd/runtime` facade already exposes checked vector,
SGEMM/GEMV, softmax, layer norm/RMSNorm, RoPE, activation, and BF16 helpers
with AVX/NEON/RVV capability reporting. Z-Image-specific kernels should be added
there only as public checked entrypoints with scalar fallback, keeping private
assembly under backend-owned packages.

## Work breakdown

1. **Inspection/readiness** — implemented via `loader/config/zimage.go` and
   `cmd/image/zimageinspect`. This validates the Diffusers component graph and reports
   an explicit `runtime_ready=false` until generation exists.
2. **Text conditioning** — reuse/adapt Qwen tokenizer and Qwen3 encoder support
   for hidden-state conditioning, without causal generation/KV assumptions.
3. **S3-DiT block reference** — implement timestep/text/image-token embedding,
   single-stream attention, MLP, normalization/modulation, and refiner blocks.
4. **FlowMatch Euler scheduler** — implement timestep schedule and latent update
   parity fixtures.
5. **AutoencoderKL decode** — implement convolution/residual/attention decode
   path from 16-channel latents to RGB.
6. **SIMD acceleration** — promote hot reference ops to checked SIMD APIs and
   add AVX/NEON/RVV assembly where profiling shows benefit.
7. **End-to-end fixtures** — pin a tiny prompt/seed/step fixture against a
   trusted reference before claiming generation support.

## Current status

Inspection only. The code can recognize and summarize the published
Z-Image-Turbo layout, but image generation is intentionally not marked ready
until the S3-DiT, scheduler, and VAE paths have native Go/SIMD implementations.
