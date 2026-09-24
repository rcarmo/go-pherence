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
4. **FlowMatch Euler scheduler** — the default four-step, weight-free schedule
   and one latent update have an independent numerical fixture. Custom schedules
   and released-model denoising are not covered.
5. **AutoencoderKL decode** — the model-free 16-channel NCHW shape and
   `(latents / scaling_factor) + shift_factor` decode-input boundary have a
   pinned NumPy float32 fixture. Convolution/residual/attention decode to RGB
   remains unimplemented.
6. **SIMD acceleration** — promote hot reference ops to checked SIMD APIs and
   add AVX/NEON/RVV assembly where profiling shows benefit.
7. **End-to-end fixtures** — pin a tiny prompt/seed/step fixture against a
   trusted reference before claiming generation support.

## Current status

Inspection plus a weight-free FlowMatch Euler slice are implemented. The
scheduler uses the pipeline's descending default sigma list, applies the pinned
scheduler shift of `3.0`, converts to 1,000-step timesteps and appends a terminal
zero sigma. `model/zimage/testdata/flowmatch_reference.json` records a four-step
schedule and first Euler update from the pinned Diffusers formulas, evaluated
independently using NumPy float32. The source revision and SHA-256 hashes are in
the fixture. Tests do not execute the full Diffusers pipeline or load model
weights. No independent released-model denoising parity has been run.

The pinned VAE config has four `block_out_channels`, so the pipeline accepts
image dimensions divisible by 16 and prepares 16-channel NCHW latents at one
eighth the image height and width. `model/zimage/testdata/vae_boundary_reference.json`
records the decode-input arithmetic and provenance. Its synthetic float32
values test preprocessing only; they do not exercise an AutoencoderKL network.

`loader/config/zimage.go` still reports `runtime_ready=false`: Qwen3 text
conditioning, S3-DiT and AutoencoderKL image decode are not implemented.
Neither model-free slice can generate images.
