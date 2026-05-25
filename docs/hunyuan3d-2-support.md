# Hunyuan3D-2 support assessment

This note maps Tencent Hunyuan3D-2 onto the current `go-pherence` codebase and identifies the work required before native support is realistic.

## Source reviewed

- Upstream code: <https://github.com/Tencent-Hunyuan/Hunyuan3D-2>
- Primary model repos:
  - `tencent/Hunyuan3D-2`
  - `tencent/Hunyuan3D-2mini`
  - `tencent/Hunyuan3D-2mv`
  - `tencent/Hunyuan3D-2.1`
- Main upstream entry point: `hy3dgen.shapegen.Hunyuan3DDiTFlowMatchingPipeline`
- Texture path: `hy3dgen.texgen.Hunyuan3DPaintPipeline`

The local review clone used for this assessment was placed under `/workspace/tmp/Hunyuan3D-2-info` and should be treated as disposable reference material, not vendored source.

## Executive summary

Hunyuan3D-2 support is a new model family and pipeline class, not a minor architecture switch in the current LLM runtime. The nearest existing pieces in `go-pherence` are useful but incomplete:

- safetensors/config loading already exists;
- tensor primitives, Conv1D, pooling, LayerNorm/GELU, matmul, and SIMD/NVIDIA backends are reusable;
- recent Whisper/audio work proves non-LLM frontends can be added;
- there is no native diffusion pipeline abstraction yet;
- there is no native DINO/CLIP vision transformer frontend yet;
- there is no native Hunyuan3D DiT implementation yet;
- there is no native ShapeVAE volume decoder or marching-cubes mesh output path yet;
- Hunyuan3D texture generation is a much larger Stable-Diffusion-family subproject and should not be the first target.

Recommended first milestone: **image-to-shape only, no texture, no turbo FlashVDM, CPU reference first, one standard Hunyuan3D-2mini fixture.** The standard `hunyuan3d-dit-v2-mini` target is preferred over `mini-turbo` for initial parity because it keeps the first fixture on the non-turbo flow path while retaining the smaller 0.6B-class DiT and 512-latent VAE footprint.

## What Hunyuan3D-2 actually runs

The upstream shape path is approximately:

1. Load/recenter/resize an RGBA/RGB input image.
2. Encode image condition with a frozen vision encoder:
   - `DinoImageEncoder`, `CLIPImageEncoder`, or dual/multiview variants depending on model config.
   - The common v2.0 single-view config uses DINO features as `contexts['main']`.
3. Sample latent tensor with shape from `vae.latent_shape`.
4. Run a flow-matching diffusion loop:
   - scheduler: `FlowMatchEulerDiscreteScheduler`;
   - default flow pipeline uses sigma schedule from `0 -> 1` over `num_inference_steps`;
   - denoiser: `Hunyuan3DDiT`;
   - classifier-free guidance duplicates latents/conditions and blends conditional/unconditional predictions.
5. Decode final latents with `ShapeVAE`:
   - VAE transformer/geo decoder produces volume logits;
   - surface extraction uses marching cubes by default (`mc`) or optional differentiable marching cubes (`dmc`).
6. Export a mesh (`trimesh`) as GLB/OBJ/etc. in Python.

The shape model config for `tencent/Hunyuan3D-2/hunyuan3d-dit-v2-0` includes:

```text
model.target: hy3dgen.shapegen.models.Hunyuan3DDiT
in_channels: 64
context_in_dim: 1536
hidden_size: 1024
num_heads: 16
depth: 16
depth_single_blocks: 32
axes_dim: [64]
theta: 10000
qkv_bias: true

vae.target: hy3dgen.shapegen.models.ShapeVAE
num_latents: 3072
embed_dim: 64
width: 1024
heads: 16
num_decoder_layers: 16
qk_norm: true
scale_factor: 0.9990943042622529

conditioner.target: hy3dgen.shapegen.models.SingleImageEncoder
conditioner.main_image_encoder.type: DinoImageEncoder
scheduler.target: hy3dgen.shapegen.schedulers.FlowMatchEulerDiscreteScheduler
```

Published shape checkpoints are safetensors under subfolders such as:

```text
hunyuan3d-dit-v2-0/config.yaml
hunyuan3d-dit-v2-0/model.fp16.safetensors
hunyuan3d-dit-v2-0/model.safetensors
hunyuan3d-dit-v2-0-fast/model.fp16.safetensors
hunyuan3d-dit-v2-0-turbo/model.fp16.safetensors
hunyuan3d-dit-v2-mini*/...
hunyuan3d-dit-v2-mv*/...
```

The texture path pulls in Diffusers/Stable-Diffusion-style pipelines, ControlNet/IP-Adapter/de-lighting/super-resolution utilities, UV unwrapping/rasterization (`xatlas`, custom rasterizers), and should be treated as a separate later project.

## Fit against current go-pherence

### Reusable now

- `loader/safetensors` can read tensor payloads and metadata.
- `loader/config` is already split enough to add a YAML/config adapter or a Hunyuan-specific config parser.
- `tensor` has checked tensor shape helpers and NN-style operations (`Linear`, `Conv1D`, pooling, LayerNorm/GELU-style building blocks) useful for a CPU reference path.
- `backends/simd/runtime` and `backends/nvidia/runtime` already expose several hot kernels that map to attention/MLP/matmul-heavy DiT code.
- `model` has patterns for architecture-specific packages (`model/qwen`, `model/gemma4`) and diagnostic fixtures.
- `loader/audio` + Whisper work shows the repo can host non-token frontends without forcing everything through LLM generation.

### Missing or insufficient

- **Diffusion pipeline abstraction.** Current generation code is token/KV-cache oriented. Hunyuan3D needs iterative latent denoising, scheduler state, CFG duplication/blending, and image-conditioned contexts.
- **Vision transformer frontend.** Hunyuan3D conditioning depends on DINOv2/CLIP vision encoders and torchvision-like image transforms. There is no DINO/CLIP ViT model package yet.
- **Hunyuan3D DiT blocks.** The denoiser uses double-stream image/text blocks, single-stream blocks, adaLN modulation, q/k RMSNorm, and attention over concatenated condition/latent streams. This is closer to Flux/DiT than to LLaMA/Qwen/Gemma decode.
- **ShapeVAE and volume decoding.** The VAE is a transformer-based latent-to-volume model with Fourier embeddings, cross-attention decoder, chunked volume evaluation, and surface extraction.
- **Marching cubes/mesh output.** Go needs a mesh package or exporter path for vertices/faces/GLB/OBJ. Python uses `skimage.measure.marching_cubes` and `trimesh`.
- **YAML/config loading.** Hunyuan3D config is YAML with Python target strings; go-pherence currently focuses on JSON model configs.
- **Texture generation dependencies.** Native texture support would require SD/ControlNet/IP-Adapter-like image diffusion, UV unwrapping/rasterization, and mesh texture baking.

## Proposed implementation plan

### Phase H0 — Research fixture and inventory

Goal: make Hunyuan3D support measurable without downloading every model variant.

- Add `docs/hunyuan3d-2-support.md` (this file) and keep it current.
- [x] Add a small helper script or command to inspect Hugging Face Hunyuan subfolders and print:
  - config path;
  - safetensors files;
  - tensor names/shapes/dtypes;
  - top-level checkpoint key groups (`model`, `vae`, `conditioner`).
- [x] Decide the first target:
  - `tencent/Hunyuan3D-2mini`, subfolder `hunyuan3d-dit-v2-mini`.
  - Rationale: smallest single-view standard/non-turbo shape target; avoids coupling the initial fixture to turbo/FlashVDM behavior.
- [ ] Generate Python golden fixtures:
  - [x] preprocessed image tensor fixture generator (`scripts/hunyuan3d_image_fixture.py`, dependency-gated on `numpy`, `Pillow`, and `opencv-python`);
  - [ ] DINO/CLIP condition embeddings (`scripts/hunyuan3d_conditioner_fixture.py` is available, but requires a local Hunyuan3D checkout plus Torch/Transformers/Safetensors image dependencies and local checkpoint payloads);
  - [x] scheduler sigmas/timesteps via `scripts/hunyuan3d_fixture_inventory.py`, with Go parity helpers in `loader/config`;
  - [ ] one denoiser step output for fixed seed/latents (`scripts/hunyuan3d_denoiser_fixture.py` is available, but requires local Hunyuan3D deps and checkpoint payloads);
  - [ ] final latent for a very low step count (`scripts/hunyuan3d_lowstep_latent_fixture.py` is available, but requires local Hunyuan3D deps and checkpoint payloads);
  - [ ] optional low-resolution mesh/volume logits (`scripts/hunyuan3d_mesh_fixture.py` is available, but requires local Hunyuan3D deps and checkpoint payloads).

Acceptance:

- Metadata inventory command identifies all required tensor groups and dimensions.
- Golden fixture can be reproduced with Python and local cached weights.

Current status: `scripts/hunyuan3d_check_fixture_env.py` reports which optional Python dependencies and local files are present before attempting heavy fixture generation. `scripts/hunyuan3d_fixture_inventory.py` can generate a metadata fixture from Hugging Face without downloading full tensor payloads. With `--include-tensors`, it fetches only safetensors header bytes. It also emits a small FlowMatch scheduler reference (`sigmas`, timesteps, normalized model timestep inputs, and Euler step formula) for early Go parity tests. `scripts/hunyuan3d_image_fixture.py` mirrors upstream `ImageProcessorV2` preprocessing for a file or synthetic RGBA input and emits tensor summaries/hashes; it is optional because it requires local Python image dependencies. `scripts/hunyuan3d_conditioner_fixture.py` can emit compact DINO/CLIP conditioner embedding hashes/shapes once a local Hunyuan3D Python environment and checkpoint payloads are available. `scripts/hunyuan3d_denoiser_fixture.py` uses the same local environment to instantiate the conditioner and DiT model, generate fixed-seed latents, and emit compact one-step denoiser summaries. `scripts/hunyuan3d_lowstep_latent_fixture.py` runs the upstream FlowMatch loop for a small number of steps and emits initial/final latent summaries without VAE decode or mesh export. `scripts/hunyuan3d_mesh_fixture.py` extends that local-only path through VAE decode and low-resolution marching-cubes mesh summaries. For `tencent/Hunyuan3D-2mini/hunyuan3d-dit-v2-mini`, the header inventory reports one `model.fp16.safetensors` file with `model`, `vae`, and `conditioner` tensor groups.

### Phase H1 — Loader/config plumbing

Goal: load Hunyuan3D shape checkpoints without running inference.

- [x] Add `loader/config/hunyuan3d.go` for YAML-derived shape config structs.
- [x] Add safetensors-style tensor-name grouping for Hunyuan files that split keys by `model.*`, `vae.*`, `conditioner.*`.
- [x] Add `model/hunyuan3d` package with static config structs and shape validation.
- [x] Add `cmd/hy3dinspect` or extend existing inspect commands to print Hunyuan3D config and tensor coverage.

Acceptance:

- Local mini checkpoint loads metadata and tensor manifests.
- Unit tests cover config parsing, tensor group binding, and malformed shape errors.

Current status: YAML config parsing and tensor-name inventory helpers are present under `loader/config` with unit tests. `model/hunyuan3d` contains static shape metadata validation, latent-shape helpers, required tensor-group coverage checks, fixture-compatible float32 tensor summaries/hashes, and a deterministic Go `ImageProcessorV2` preprocessing parity target for alpha-mask recentering, white-background compositing, square resize, and BCHW `[-1, 1]` tensors. `cmd/hy3dinspect` / `make hunyuan3d-inspect` can print validated config shape metadata and enforce optional safetensors group coverage for local checkpoints. These deliberately stop at metadata/config coverage and do not claim runtime support.

### Phase H2 — Image preprocessing and conditioner

Goal: reproduce condition embeddings for one fixture.

- [x] Add Python fixture generator for upstream `ImageProcessorV2`:
  - RGBA mask handling;
  - recenter with border ratio;
  - resize to model image size;
  - normalize to `[-1, 1]` BCHW tensor range.
- [x] Implement Go image preprocessing equivalent to upstream `ImageProcessorV2`:
- Implement or import a minimal DINOv2/CLIP-ViT encoder path:
  - patch embedding;
  - class token/position embeddings;
  - transformer encoder blocks;
  - LayerNorm/GELU/attention parity.
- Start with CPU F32/BF16 reference; GPU later.

Acceptance:

- Go image preprocessing matches Python fixture within tolerance.
- Go condition embeddings match Python for a frozen tiny/small fixture.

### Phase H3 — Hunyuan3D-DiT CPU reference

Goal: one denoiser forward pass matches Python.

Implement the `Hunyuan3DDiT` subset:

- timestep sinusoidal embedding + MLP embedder;
- latent input projection;
- condition input projection;
- double-stream blocks:
  - adaLN modulation;
  - separate image/text qkv;
  - q/k RMSNorm;
  - concat attention over condition+latent streams;
  - gated residual attention and MLP;
- single-stream blocks:
  - combined qkv+MLP projection;
  - q/k RMSNorm;
  - attention + GELU MLP concat projection;
  - gated residual;
- final adaLN + projection back to latent channels.

Acceptance:

- One fixed denoiser step matches Python fixture.
- A low-step denoising loop matches Python final latents within tolerance.

### Phase H4 — Scheduler and shape-generation loop

Goal: produce final latents without mesh export.

- [x] Implement metadata-level `FlowMatchEulerDiscreteScheduler` reference helpers:
  - sigma/timestep generation;
  - `step`: `sample + (sigma_next - sigma) * model_output`;
  - flow pipeline convention where model receives timesteps normalized to `[0, 1]`.
- [x] Implement CFG duplication/blending for Hunyuan3D flow pipeline.
- [x] Add deterministic random latent generation for fixture parity.

Acceptance:

- Go final latent tensor matches Python low-step fixture for fixed seed.

### Phase H5 — ShapeVAE decode and mesh extraction

Goal: output an actual mesh for image-to-shape.

- Implement enough `ShapeVAE` decode path to produce volume logits:
  - post-KL projection;
  - VAE transformer;
  - Fourier embedding;
  - cross-attention geo decoder;
  - chunked volume decoder.
- Add a marching-cubes implementation or dependency review:
  - preferred: pure Go implementation with deterministic tests;
  - fallback: optional external tool path only if kept outside default static build.
- Add mesh type and OBJ/PLY/GLB exporter decisions.

Acceptance:

- Low-resolution volume/mesh fixture matches Python topology/geometry within tolerance.
- CLI can write an OBJ/PLY/GLB for a local image.

### Phase H6 — GPU acceleration

Goal: make Hunyuan3D usable on local RTX 3060-class hardware.

- Reuse existing NVIDIA matmul/attention kernels where shapes fit.
- Add batched DiT attention and MLP scheduling for sequence lengths around:
  - condition tokens: DINO/CLIP patch sequence;
  - latent tokens: `num_latents` (e.g. 3072);
  - hidden size: 1024;
  - heads: 16.
- Treat VAE volume decode as chunked GPU work; keep marching cubes CPU initially.

Acceptance:

- One mini shape generation path completes within practical time on RTX 3060.
- CPU and GPU denoiser outputs remain within agreed tolerances.

### Phase H7 — Texture and multiview later

Only after image-to-shape works:

- `Hunyuan3D-Paint`/texture is a Stable-Diffusion-family pipeline, not just a mesh postprocess.
- Multiview shape uses `DinoImageEncoderMV` and view embeddings; add after single-view.
- Turbo/FlashVDM changes VAE/decoder behavior; add after standard VAE parity.

## Recommended next step

Do **not** add a CLI that claims runtime support yet. The fixture ladder is now scaffolded, so the next practical step is to create the local Python/checkpoint environment and run the actual fixtures:

1. `make hunyuan3d-fixture-env` to see missing optional dependencies and files.
2. Install local Hunyuan3D Python dependencies and place/cache the standard `Hunyuan3D-2mini/hunyuan3d-dit-v2-mini` config/checkpoint/image.
3. Run `make hunyuan3d-image-fixture`, `make hunyuan3d-conditioner-fixture`, `make hunyuan3d-denoiser-fixture`, and `make hunyuan3d-lowstep-fixture`.
4. Use `make hunyuan3d-mesh-fixture` only after the low-step latent fixture is stable, because VAE decode/marching-cubes is heavier and adds more dependency surface.
5. Implement the Go image preprocessing and DINO/conditioner path only after these Python goldens are stable.

This avoids prematurely coupling Hunyuan3D to the token-generation APIs and keeps the work aligned with the existing backend-first package boundaries.

## Open questions

- Which first target should be canonical: `Hunyuan3D-2mini` standard, mini-turbo, or full `Hunyuan3D-2`?
- Is CPU reference-only acceptable for early parity, even if full generation is slow?
- Should mesh export be OBJ/PLY first, with GLB later?
- Should DINO/CLIP vision encoders live under `models/vision`, `model/hunyuan3d`, or a new frontend package?
- Is texture support in scope for go-pherence, or should native shape generation be the boundary?
