# TRELLIS.2 support notes

This document tracks TRELLIS.2 research and a staged metadata/fixture plan for `go-pherence`. It is deliberately scoped to inventory and parity scaffolding first. Nothing in this document claims TRELLIS.2 runtime inference support.

## Official references

- Project page: <https://microsoft.github.io/TRELLIS.2/>
- GitHub: <https://github.com/microsoft/TRELLIS.2>
- Model: <https://huggingface.co/microsoft/TRELLIS.2-4B>
- Paper: <https://arxiv.org/abs/2512.14692>
- Demo: <https://huggingface.co/spaces/microsoft/TRELLIS.2>

## Snapshot

TRELLIS.2 is a 4B-parameter image-to-3D system for PBR textured assets. The public materials describe it as using native and compact structured latents, O-Voxel / Omni-Voxel sparse voxel representations, and sparse compression VAEs. Published examples target output resolutions from 512³ to 1536³.

Reported project-page H100 timings:

| Resolution | Shape | Texture | Total |
| --- | ---: | ---: | ---: |
| 512³ | 2s | 1s | 3s |
| 1024³ | 10s | 7s | 17s |
| 1536³ | 35s | 25s | 60s |

The model is MIT licensed on Hugging Face and the GitHub repository is also MIT licensed.

## Hardware and software constraints

The upstream README states that TRELLIS.2 is Linux-only/tested and expects an NVIDIA GPU with at least 24GB VRAM. It was verified on A100/H100, recommends CUDA 12.4, Python 3.8+, Conda, PyTorch 2.6.0 CUDA 12.4, and `flash-attn` by default. `xformers` can be used for GPUs without `flash-attn` support.

These constraints mean TRELLIS.2 should initially be treated as a metadata/fixture target for `go-pherence`, not as near-term pure-Go CPU inference work.

## Public artifact inventory

The `microsoft/TRELLIS.2-4B` model repository currently exposes JSON configs plus safetensors checkpoints:

```text
pipeline.json
texturing_pipeline.json

ckpts/shape_enc_next_dc_f16c32_fp16.json
ckpts/shape_enc_next_dc_f16c32_fp16.safetensors
ckpts/shape_dec_next_dc_f16c32_fp16.json
ckpts/shape_dec_next_dc_f16c32_fp16.safetensors

ckpts/tex_enc_next_dc_f16c32_fp16.json
ckpts/tex_enc_next_dc_f16c32_fp16.safetensors
ckpts/tex_dec_next_dc_f16c32_fp16.json
ckpts/tex_dec_next_dc_f16c32_fp16.safetensors

ckpts/ss_flow_img_dit_1_3B_64_bf16.json
ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors

ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json
ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors
ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json
ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors

ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json
ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors
ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json
ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors
```

This split suggests the first native metadata groups should be:

- shape VAE encoder
- shape VAE decoder
- texture VAE encoder
- texture VAE decoder
- sparse-structure image-conditioned flow
- structured-latent image-to-shape flow
- structured-latent image+shape-to-texture flow

## Architecture cues from upstream source

Important upstream files to study before implementing runtime work:

```text
trellis2/models/sc_vaes/fdg_vae.py
trellis2/models/sc_vaes/sparse_unet_vae.py
trellis2/models/sparse_structure_flow.py
trellis2/models/sparse_structure_vae.py
trellis2/models/structured_latent_flow.py
trellis2/pipelines/trellis2_image_to_3d.py
trellis2/pipelines/trellis2_texturing.py
trellis2/pipelines/samplers/flow_euler.py
```

Likely low-level areas:

- O-Voxel representation and sparse voxel structure
- sparse convolution and sparse attention modules
- sparse compression VAE encoders/decoders
- DiT-style flow models over sparse/structured latents
- PBR texture attributes: base color, roughness, metallic, opacity/alpha

## Staged plan

### T0 — research capture

- [x] Identify official links and model availability.
- [x] Capture artifact split, license, and hardware requirements.
- [ ] Keep this document current as upstream repo/model files change.

### T1 — metadata inventory tooling

- [ ] Add `scripts/trellis2_fixture_inventory.py`.
- [ ] Support local model directory and Hugging Face metadata mode.
- [ ] Parse `pipeline.json` and `texturing_pipeline.json`.
- [ ] Parse checkpoint JSON configs.
- [ ] Optionally fetch safetensors headers only, never payloads by default.
- [ ] Emit compact JSON inventory with checkpoint families, config summaries, tensor counts, and examples.
- [ ] Add `make trellis2-inventory`.

### T2 — Go config parsing and validation

- [x] Add `loader/config/trellis2.go`.
- [x] Add representative JSON tests for pipeline and checkpoint config snippets.
- [x] Validate required checkpoint families for shape-only and shape+texture pipelines. `loader/config` now has metadata structs, summaries, representative tests, explicit-family parsing for ambiguous `SLatFlowModel` configs, and required pipeline model-key checks.
- [x] Add compact summary helpers for checkpoint dtype, resolution, latent dimensions, and flow model family.

### T3 — parity fixtures

- [ ] Inspect upstream `flow_euler.py` and add a native Go scheduler reference if simple and stable.
- [ ] Reuse fixture-compatible tensor summary/hash helpers for TRELLIS.2 generated fixture outputs.
- [ ] Add dependency-gated Python fixture scaffolds for low-step sparse-structure and structured-latent flow outputs.

### T4 — runtime feasibility study

- [ ] Map O-Voxel metadata and sparse tensor layouts.
- [ ] Inventory sparse convolution/attention kernels.
- [ ] Identify which kernels could run on CPU, CUDA, Vulkan, or future backend abstractions.
- [ ] Defer any runtime inference support claim until metadata, fixtures, and kernel requirements are validated.

## Relationship to Hunyuan3D work

Hunyuan3D remains the active implementation track. TRELLIS.2 should start as a parallel metadata and fixture-parity track because it is public, MIT licensed, and architecturally important, but its sparse-native runtime requirements are larger than the current Hunyuan3D metadata ladder.
