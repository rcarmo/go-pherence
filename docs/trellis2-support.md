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

- [x] Inspect upstream `flow_euler.py` and add a native Go scheduler reference if simple and stable.
- [x] Reuse fixture-compatible tensor summary/hash helpers for TRELLIS.2 generated fixture outputs.
- [x] Add dependency-gated Python fixture scaffolds for low-step sparse-structure and structured-latent flow outputs.

### T4 — runtime feasibility study

- [x] Map O-Voxel metadata and sparse tensor layouts.
- [x] Inventory sparse convolution/attention kernels.
- [x] Identify which kernels could run on CPU, CUDA, Vulkan, or future backend abstractions.
- [ ] Defer any runtime inference support claim until metadata, fixtures, and kernel requirements are validated.

## O-Voxel and sparse tensor layout notes

Upstream O-Voxel lives in the separate `o-voxel/` package embedded as part of the TRELLIS.2 repository. It is not just a file format shim: it includes Python modules plus compiled C++/CUDA extension sources for conversion, rendering, post-processing, and GLB/export workflows.

Important O-Voxel files:

```text
o-voxel/o_voxel/io/npz.py
o-voxel/o_voxel/io/vxz.py
o-voxel/o_voxel/io/ply.py
o-voxel/o_voxel/convert/flexible_dual_grid.py
o-voxel/o_voxel/convert/volumetic_attr.py
o-voxel/o_voxel/postprocess/*
o-voxel/src/convert/*
o-voxel/src/render/*
```

Observed layout concepts:

- O-Voxel examples pass `coords`, voxel data/features, and rendering/export metadata around explicitly.
- Sparse coordinates are integer tensors; TRELLIS.2 sparse module comments state coords should be in `[0, 1023]`.
- `trellis2.modules.sparse.basic.SparseTensor` is the central sparse wrapper and extends `VarLenTensor`.
- `SparseTensor` stores feature rows (`feats`) and corresponding coordinate rows (`coords`), with data for the same batch expected to be contiguous.
- Coordinates include a leading batch coordinate for sparse backends; user-facing sparse structure coords are often projected as `[batch, x, y, z]` after `torch.argwhere(decoded)[:, [0, 2, 3, 4]]`.
- `SparseTensor` can wrap multiple backend layouts:
  - `torchsparse.SparseTensor`
  - `spconv.pytorch.SparseConvTensor`
  - fallback dict-style `{feats, coords}` data
- `spconv` construction derives `spatial_shape = coords.max(0)+1`, uses `spatial_shape[1:]` for spatial dimensions, and `spatial_shape[0]` as batch size.
- `VarLenTensor` carries a compact variable-length layout with sequence lengths, cumulative sequence lengths, and broadcast maps. This is relevant for sparse attention and any future Go fixture representation.

Pipeline sparse/voxel transitions seen in `trellis2_image_to_3d.py`:

1. Sparse-structure flow samples dense noise with shape:

   ```text
   [num_samples, in_channels, reso, reso, reso]
   ```

2. Sparse-structure decoder thresholds occupancy:

   ```text
   decoded = decoder(z_s) > 0
   coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
   ```

3. Shape structured-latent flow creates a `SparseTensor`:

   ```text
   feats = randn([coords.shape[0], flow_model.in_channels])
   coords = coords
   ```

4. Shape/texture decoders operate on structured sparse latents and later convert to mesh/O-Voxel/PBR asset outputs.

Near-term Go implication: store sparse fixture metadata as `{coords_shape, coords_dtype/hash, feats_shape, feats_dtype/hash, coordinate_order}` first. Avoid committing to a runtime sparse backend until `torchsparse`/`spconv` and O-Voxel conversion kernels are fully inventoried.

## Sparse kernel and backend inventory

TRELLIS.2 sparse runtime is controlled by `trellis2/modules/sparse/config.py`:

```text
SPARSE_CONV_BACKEND / config.CONV: none | spconv | torchsparse | flex_gemm
SPARSE_ATTN_BACKEND or ATTN_BACKEND / config.ATTN: xformers | flash_attn | flash_attn_3
Defaults: CONV=flex_gemm, ATTN=flash_attn
```

Sparse convolution layer dispatch:

```text
trellis2/modules/sparse/conv/conv.py
  SparseConv3d.forward -> sparse_conv3d_forward(...)
  SparseInverseConv3d.forward -> sparse_inverse_conv3d_forward(...)

trellis2/modules/sparse/conv/conv_flex_gemm.py
  flex_gemm.ops.spconv_subm / spconv / inverse-conv style CUDA kernels

trellis2/modules/sparse/conv/conv_spconv.py
  spconv.pytorch SparseConv3d/SubMConv3d/SparseInverseConv3d backend

trellis2/modules/sparse/conv/conv_torchsparse.py
  torchsparse convolution backend

trellis2/modules/sparse/conv/conv_none.py
  placeholder/no sparse convolution runtime path
```

Sparse attention layer dispatch:

```text
trellis2/modules/sparse/attention/modules.py
  SparseMultiHeadAttention

trellis2/modules/sparse/attention/full_attn.py
  sparse_scaled_dot_product_attention
  xformers variable-length attention path
  flash_attn varlen qkv-packed path
  flash_attn_3 path where available
```

Other sparse primitives used by models:

```text
trellis2/modules/sparse/linear.py       SparseLinear
trellis2/modules/sparse/spatial/basic.py SparseDownsample / SparseUpsample
trellis2/modules/sparse/subdivide.py    sparse subdivision helpers
trellis2/modules/sparse/basic.py        VarLenTensor, SparseTensor, sparse_cat, sparse_unbind
```

Model usage hotspots:

- `trellis2/models/sc_vaes/sparse_unet_vae.py`: heavy use of `SparseConv3d`, `SparseLinear`, `SparseDownsample`, `SparseUpsample`, and Sparse ConvNeXt/ResBlock VAE blocks.
- `trellis2/models/sc_vaes/fdg_vae.py`: flexible-dual-grid VAE blocks using sparse ConvNeXt components.
- `trellis2/models/structured_latent_flow.py`: `SparseLinear` and `sparse_cat` around sparse structured latents.
- DiT/flow models depend on sparse attention backends through `SparseMultiHeadAttention` and variable-length attention layouts.

Backend feasibility:

| Area | Upstream backend | Near-term Go feasibility | Notes |
| --- | --- | --- | --- |
| Sparse conv3d / inverse conv | `flex_gemm`, `spconv`, `torchsparse` | Low without CUDA/custom sparse kernels | Core VAE/decoder blocker. CPU scalar fallback would be correctness-only and likely too slow. |
| Sparse attention | `flash_attn`, `flash_attn_3`, `xformers` | Low without varlen attention kernels | Requires packed QKV, cumulative sequence lengths, and GPU-friendly varlen attention. |
| Sparse linear / MLP | PyTorch dense matmul over sparse feature rows | Medium | Could map to existing GEMM/GEMV once feature layouts are fixed. |
| Sparse cat/unbind/layout | Python tensor bookkeeping | High for metadata/fixtures | Good first Go target for fixture validation. |
| Sparse down/up-sample/subdivide | custom sparse coordinate transforms | Medium | Implementable as coordinate/layout transforms before conv kernels. |
| O-Voxel conversion/render/postprocess | `o-voxel` C++/CUDA extensions | Low for runtime, medium for metadata | Start with file/metadata inspection only. |
| Mesh/GLB export | O-Voxel + rendering/postprocess stack | Low in pure Go initially | Keep out of runtime claims. |

CPU path: viable only for metadata, fixtures, small coordinate transforms, sparse linear math, and correctness micro-fixtures. Full TRELLIS.2 inference is not CPU-practical without a much larger sparse-kernel project.

CUDA path: closest to upstream because `flex_gemm`, `flash_attn`, `xformers`, `spconv`, O-Voxel render/convert, and nvdiffrast/nvdiffrec are CUDA-oriented. A Go CUDA backend would still need either bindings or native kernels for sparse conv and varlen attention.

Vulkan path: speculative. Dense matmul/MLP pieces are plausible, but sparse conv3d, inverse conv, varlen attention, O-Voxel conversion, and differentiable/rendering kernels would require substantial new compute kernels.

Future backend abstraction: separate sparse layout/coordinate transforms from kernel execution. First stable Go surfaces should be fixture structures and validators for `SparseTensor`-like `{coords, feats, layout}` data, not inference kernels.

## Relationship to Hunyuan3D work

Hunyuan3D remains the active implementation track. TRELLIS.2 should start as a parallel metadata and fixture-parity track because it is public, MIT licensed, and architecturally important, but its sparse-native runtime requirements are larger than the current Hunyuan3D metadata ladder.
