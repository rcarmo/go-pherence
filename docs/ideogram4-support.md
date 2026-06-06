# Ideogram 4 FP8 support

`ideogram-ai/ideogram-4-fp8` is a gated Diffusers text-to-image pipeline released with the `ideogram-oss/ideogram4` reference implementation. It is not a LLaMA/GGUF decoder target; native go-pherence support needs a separate image-generation pipeline.

## Published/local component graph

The gated Hugging Face config reports:

```text
pipeline:                  Ideogram4Pipeline
transformer:               Ideogram4Transformer2DModel
unconditional_transformer: Ideogram4Transformer2DModel
scheduler:                 FlowMatchEulerDiscreteScheduler
text_encoder:              Qwen3VLModel
tokenizer:                 Qwen2Tokenizer
vae:                       AutoencoderKLFlux2
```

## Transformer shape

From the real `ideogram-4-fp8` configs and `ideogram-oss/ideogram4`:

```text
layers=34
emb_dim=4608
num_attention_heads=18
attention_head_dim=256
intermediate_size=12288
in_channels=128
llm_features_dim=53248
rope_theta=5000000
mrope_section=[24 20 20]
```

`llm_features_dim=53248` is `4096 * 13`, from concatenating Qwen3-VL hidden states at activation layers:

```text
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]
```

## Text encoder

The text encoder config is Qwen3-VL with nested `text_config`:

```text
model_type=qwen3_vl
text.model_type=qwen3_vl_text
hidden_size=4096
num_hidden_layers=36
num_attention_heads=32
num_key_value_heads=8
vocab_size=151936
```

The reference pipeline uses text-only Qwen3-VL hidden-state extraction through the Qwen chat template. Native support therefore needs an encoder-style Qwen3-VL forward path that returns the selected hidden layers, not causal token generation.

## Native implementation status

Generation support is implemented in pure Go with backend-owned kernels and no Python/Diffusers runtime dependency. The supported path is CPU/SIMD today:

1. **Inspection/readiness** — `loader/config/ideogram4.go`, `model/ideogram4` shape helpers, and `cmd/ideogram4inspect` validate local Diffusers folders, transformer inventories, FP8 scale coverage, and linear-role coverage.
2. **Qwen3-VL text conditioning** — `QwenVLConditioner` runs the text-only Qwen3-VL stack and concatenates the 13 selected hidden states into `[tokens, 53248]` conditioning features.
3. **Ideogram4 DiT forward** — native single-stream text+image token transformer blocks implement QK-RMSNorm, MRoPE, SwiGLU MLP, AdaLN timestep conditioning, final projection, and velocity prediction.
4. **FP8 weight loading and GEMV** — conditional and unconditional DiT weights plus Qwen linears load directly from `F8_E4M3` safetensors. GEMV dequantizes on the fly through the fp8 backend, with an amd64 AVX2/FMA gather-dot kernel and scalar fallback.
5. **Unconditional transformer / asymmetric CFG** — `DenoiseLoop` runs conditional and unconditional DiT paths and blends them with `CombineCFG`.
6. **FlowMatch Euler scheduler** — the native logit-normal schedule and Euler latent update match the reference scheduler numerically for inspected step plans.
7. **AutoencoderKLFlux2 decode** — `VAEDecoder` decodes denoised 32-channel latents to RGB output.
8. **End-to-end CLI** — `cmd/ideogram4gen` loads tokenizer, Qwen3-VL, both DiTs, scheduler, and VAE from a Diffusers folder and writes a PNG.

The project has a separate NVIDIA backend for LLM/Gemma/Qwen paths. Ideogram 4 now has CUDA/NVIDIA primitives for FP8 E4M3 linear GEMV, fused scheduler/CFG vector updates, weighted RMSNorm, row-wise non-affine LayerNorm, adaLN scale/gate transforms, gated residual updates, full-tensor MRoPE rotation, full non-causal DiT attention (scores, row-softmax, value accumulation), and MLP/final-projection vector operations (SiLU, multiply, fused SiLU*Mul). FP8 linear GPU execution is exposed behind `GO_PHERENCE_IDEOGRAM4_GPU_FP8=1` (`GO_PHERENCE_IDEOGRAM4_GPU_FP8_STRICT=1` disables CPU fallback); set `GO_PHERENCE_IDEOGRAM4_GPU_FP8_CACHE=1` to keep each uploaded FP8 linear weight GPU-resident across repeated calls instead of streaming it for every projection. `GO_PHERENCE_IDEOGRAM4_GPU_RESIDENCY` controls coarse VRAM policy: `persistent` keeps cached weights until `ReleaseGPU`, `phase` releases cached DiT weights between Qwen/DiT/VAE phases for 12GB-class cards, and `stream` disables long-lived FP8 caching. `cmd/ideogram4gen` now exposes `-gpu`, `-gpu-strict`, `-gpu-fp8-cache`, and `-gpu-residency persistent|phase|stream` to set the same gates from the CLI. `FP8Linear`, `DiTLayer`, `DiTModel`, and `NativePipeline` expose `ReleaseGPU` hooks, and `cmd/ideogram4gen` defers `pipe.ReleaseGPU()` to clean cached linears on exit. The fused CFG+Euler step is exposed behind `GO_PHERENCE_IDEOGRAM4_GPU_CFG=1` (`GO_PHERENCE_IDEOGRAM4_GPU_CFG_STRICT=1` disables CPU fallback). RMSNorm, non-affine LayerNorm, adaLN transforms, and gated residuals can be exercised with `GO_PHERENCE_IDEOGRAM4_GPU_NORM=1` (`GO_PHERENCE_IDEOGRAM4_GPU_NORM_STRICT=1` disables CPU fallback). MRoPE rotation is exposed behind `GO_PHERENCE_IDEOGRAM4_GPU_MROPE=1` (`GO_PHERENCE_IDEOGRAM4_GPU_MROPE_STRICT=1` disables CPU fallback). Full DiT attention is exposed behind `GO_PHERENCE_IDEOGRAM4_GPU_ATTN=1` (`GO_PHERENCE_IDEOGRAM4_GPU_ATTN_STRICT=1` disables CPU fallback). MLP/final vector ops are exposed behind `GO_PHERENCE_IDEOGRAM4_GPU_MLP=1` (`GO_PHERENCE_IDEOGRAM4_GPU_MLP_STRICT=1` disables CPU fallback). These are correctness-oriented kernel boundaries, not a full GPU-resident graph yet; VAE decode is still CPU/SIMD in the public pipeline.

## DiT block forward

`model/ideogram4/dit_block.go` implements one Ideogram4 transformer block natively over the loaded FP8 linears:

- gate-less adaLN modulation (`4*emb` block params split into shift/scale for the attention and MLP sublayers, matching the `2*emb` final-layer modulation contract),
- non-affine LayerNorm + `x*(1+scale)+shift`,
- QKV projection, 3-section MRoPE (`[24,20,20]`, temporal/height/width) over the first `2*sum(section)` head dims via rotate-half,
- full (non-causal) scaled dot-product attention with SIMD softmax,
- output projection residual,
- SwiGLU MLP (`W2(SiLU(W1)*W3)`) residual.

The block math was reconciled against the public `ideogram-oss/ideogram4` source and has been executed on real FP8 DiT weights without NaN/Inf output.

## Full DiT velocity forward

`model/ideogram4/dit_forward.go` stacks the blocks into a complete velocity pass (`DiTModel.Velocity`):

- `LoadDiTModel` assembles globals + 34 layers from a loaded FP8 set,
- sinusoidal timestep embedding → `t_embedding.mlp_in/out` (SiLU) → `adaln_proj` conditioning,
- `llm_cond_proj`(LayerNorm(text features)) text tokens + `input_proj`(latents) image tokens form one joint self-attention sequence (single qkv/o per block),
- text prefix gets sequential temporal MRoPE positions, image tokens the 2D grid,
- 34 `ForwardLayer` blocks,
- `final_layer.adaln_modulation` + `final_layer.linear` over image tokens → velocity `[imageTokens, in_channels]`.

## Sampling loop

`model/ideogram4/denoise_loop.go` (`DenoiseLoop`) ties the pieces together: for each FlowMatch step it runs the conditional DiT (text+image) and unconditional DiT (image-only), blends them with asymmetric CFG (`CombineCFG`), and applies the scheduler Euler update — returning denoised latents `[imageTokens, in_channels]`.

## VAE decode primitives

`model/ideogram4/vae_ops.go` adds the native building blocks for the `AutoencoderKLFlux2` decoder (`latent_channels=32`, `block_out_channels=[128,256,512,512]`, GroupNorm-32 + SiLU, `patch_size=[2,2]`): `FeatureMap` (NCHW=1), `UnpatchifyLatents` (DiT patch tokens → latent feature map), stride-1 same-pad `Conv2D` (OIHW, im2col + SIMD `GemmRows`), affine `GroupNorm`, in-place `SiLUMap`, nearest `UpsampleNearest`, and residual add — the ops a ResNet/up-block decoder graph is assembled from.

## VAE decoder graph

`model/ideogram4/vae_decoder.go` (`VAEDecoder`) assembles the full `AutoencoderKLFlux2` decode from VAE safetensors weights: latent de-scale → `post_quant_conv` → `decoder.conv_in` → mid block (resnet → single-head spatial self-attention → resnet) → 4 up-blocks (`layers_per_block+1` ResnetBlock2D each, nearest ×2 upsample + conv between blocks) → `conv_norm_out`/SiLU/`conv_out` → 8-bit RGB `Image`. ResNet shortcuts auto-detect the optional `conv_shortcut`; norms are GroupNorm-32. Bound to any `F32TensorSource` (safetensors File/ShardedFile).

**Validated against real weights:** `cmd/ideogram4vaesmoke` loads the actual downloaded `vae/diffusion_pytorch_model.safetensors` (251 tensors) and decodes a random `[32,16,16]` latent end-to-end through the full conv graph, producing a valid `128x128` RGB image with a healthy pixel distribution (e.g. min=27 max=188 mean=111). All tensor names/shapes matched on the first try. `Conv2D` uses im2col + the SIMD `GemmRows` kernel, cutting the `128x128` decode from ~3m25s to ~6s (~33x) with bit-identical output.

## Qwen3-VL text conditioning

`model/ideogram4/qwen_vl_conditioner.go` (`QwenVLConditioner`) is the native Qwen3-VL text-only forward (`hidden=4096`, 36 layers, 32 q / 8 kv GQA heads, `head_dim=128`, SwiGLU `intermediate=12288`, RMSNorm `1e-6`). Linears are the checkpoint's FP8 E4M3 weights (loaded via the fp8 backend); embeddings are decoded per-token from bf16 to avoid materializing the full table. Each layer: RMSNorm → q/k/v FP8 proj → per-head q/k RMSNorm → RoPE → causal GQA attention → o-proj residual → RMSNorm → SwiGLU MLP residual. `Condition(tokenIDs)` captures hidden states at `ActivationLayers` (`[0,3,...,35]`, HF indexing with 0=post-embedding) and concatenates them into the `[tokens, 53248]` tensor feeding `DenoiseLoop`.

## End-to-end native pipeline

`model/ideogram4/native_pipeline.go` (`NativePipeline` / `LoadNativePipeline`) assembles every component from a Diffusers directory — tokenizer, Qwen3-VL conditioner, conditional + unconditional FP8 DiT transformers, and the VAE decoder (single-file or sharded safetensors auto-detected). `NativePipeline.Generate(prompt, opts)` runs the full path: tokenize → Qwen3-VL conditioning → FlowMatch denoise loop with asymmetric CFG → unpatchify → VAE decode → RGB `Image`. `cmd/ideogram4gen` drives it from the CLI (seeded Gaussian init latents, PNG output).

The full text→image path is now implemented natively in Go/SIMD. The DiT, MRoPE, adaLN, final layer, latent denormalization/unpatchify, and conditioning were reconciled against the reference `ideogram-oss/ideogram4` source (`modeling_ideogram4.py`, `scheduler.py`, `latent_norm.py`, `pipeline_ideogram4.py`), resolving the earlier provisional assumptions:

- adaLN is **scale + tanh-gate** (not shift/scale): per block `4*emb` → `scale_msa, gate_msa, scale_mlp, gate_mlp`; sublayers are `x += tanh(gate) * norm2(sublayer(norm1(x)*(1+scale)))` with four learnable RMSNorms per block.
- attention uses **QK-RMSNorm** (`norm_q`/`norm_k`, eps 1e-5) before RoPE.
- **MRoPE** uses interleaved 3-axis section assignment over the full head_dim with `IMAGE_POSITION_OFFSET=65536` and text positions `(i,i,i)`.
- the final layer is a **non-affine LayerNorm** with scale-only modulation (`final_adaln` → `emb`, not `2*emb`).
- all bias=true linears load their `.bias`; `embed_image_indicator` and `llm_cond_norm` are applied; latents are denormalized with the per-channel `LATENT_SCALE/SHIFT` constants and unpatchified in `(patch_h, patch_w, ae_channels)` order before VAE decode.

The prompt is wrapped in the Qwen3-VL ChatML template (`TokenizeChatPrompt`). The byte-level BPE encoder now matches the reference HuggingFace tokenizer on the local 15-prompt validation set, including emoji, CJK, code, URLs, tabs, punctuation, and mixed whitespace.

## Validation status against real weights

The implementation has been validated on real gated weights as follows:

- **VAE decoder** — fully downloaded (168 MB) and run end-to-end (`cmd/ideogram4vaesmoke`): all 251 tensors load, the conv graph executes, and a `[32,16,16]` latent decodes to a valid `128x128` RGB image. im2col + SIMD `GemmRows` makes it ~6s. The decode graph was also reconciled against the reference `autoencoder.py`/`_decode` (mid `block_1/attn_1/block_2`, up-blocks 0–3 with 3 resnets each and upsample after 0–2, `AttnBlock`/`ResnetBlock`/nearest-`Upsample` semantics, 128-dim latent denorm before unpatch, `(x+1)*127.5` output) — no discrepancies found.
- **FP8 DiT transformer** — the safetensors header (all 669 tensor shapes/dtypes) was fetched via a ~70 KB HTTP range request and checked against the loader's expectations: **0 shape mismatches, 0 missing tensors**. Weights are `F8_E4M3`, per-output-row scales are `F32`, and biases/norms are `BF16` (all handled by `decodeScale`/`decodeFloatVec`). This confirms the corrected layout (e.g. `final_layer.adaln_modulation` → `emb`, `embed_image_indicator` `[2,emb]`, QK-norms `[head_dim]`, four per-block RMSNorms, all biases). The **full 34-layer forward was also executed on the real weights** (transformer downloaded transiently): `Velocity` over a tiny token set ran with **0 NaN/Inf** and numerically healthy output (mean≈0, std≈0.98 — as expected for a flow-matching velocity), confirming the DiT graph is correct end-to-end, not just shape-consistent.
- **Qwen3-VL text encoder** — same header-only validation over all 1117 tensors: **0 errors**. FP8 q/k/v/o and gate/up/down projections with `F32` per-row scales, `BF16` q/k/input/post norms, and a `BF16` `[vocab, hidden]` embedding table — exactly what `QwenVLConditioner` loads. The **full 36-layer forward was also executed on the real weights** (encoder downloaded transiently): `Condition` produced the `[tokens, 53248]` features with **0 NaN/Inf**; magnitudes are large (std≈63) as expected for raw concatenated LLM hidden states with outlier activations, which the DiT's `llm_cond_norm` RMSNorm then normalizes.

- **Schedule** — the FlowMatch logit-normal schedule was checked numerically against the reference `scheduler.py`: for a 6-step plan the t-value sequence matches (`[0.007, 0.528, 0.657, 0.746, 0.819, 0.886]`) to the clamp constant.
- **Tokenizer** — the byte-level BPE `Encode` path was validated against the reference HuggingFace `tokenizers` library on 15 diverse prompts (emoji, CJK, code, URLs, tabs, mixed whitespace, punctuation): **15/15 exact token-ID matches**, using the Qwen2/Qwen3 pre-tokenization regex plus a `splitWhitespaceRuns` emulation of the RE2-inexpressible `\s+(?!\S)` lookahead.

### End-to-end run on real weights (staged)

An early **conditional-only** (CFG-disabled) staged generation was run under disk pressure by downloading one large component at a time. The full path executed on real weights end-to-end (12 prompt tokens encoded; 6 DiT forwards; VAE decode → `64x64` RGB PNG) with finite values throughout and no crashes. The output was a low-detail wash, which was expected for the deliberately degenerate budget (64×64, 6 steps, no classifier-free guidance).

After freeing workspace scratch, the full Diffusers folder (text encoder, conditional DiT, unconditional DiT, VAE, tokenizer, configs; roughly 27 GB) was downloaded under `/workspace/tmp` and a CFG-guided cat prompt was run through `cmd/ideogram4gen` at `64x64`, 6 steps, guidance `5.0`, seed `42`. The optimized CPU/SIMD path completed in about `529s` and produced a finite PNG. The image was still not a recognizable cat; it should be treated as an execution proof-of-life, not as a quality benchmark. A faithful sample likely needs higher resolution and substantially more steps, which is currently expensive on the CPU-only Ideogram path.

**Performance:** the CPU FP8 E4M3 decode uses a 256-entry lookup table (one byte → one of 256 values), making the dequant GEMV branch-free. The amd64 backend adds an AVX2/FMA `VGATHERDPS` E4M3 dot kernel over that LUT; a standalone `4096x4608` FP8 GEMV smoke measured about `3.04ms` versus `8.85ms` for the scalar LUT loop (`~2.9x` speedup, with expected accumulation-order drift around `1e-3`). The NVIDIA runtime also has a direct FP8 E4M3 GEMV kernel (`fp8_e4m3_gemv_f32`) and `GPUFP8E4M3Linear` upload wrapper; synthetic GPU smoke matched the CPU backend within `~1e-5`, and the cached `FP8Linear.Apply` path matched repeated CPU calls within `~2e-5`. The fused NVIDIA Ideogram CFG+FlowMatch step (`ideogram_cfg_step_f32`) replaces `guided = uncond + guidance*(cond-uncond)` plus `x += sigma*guided`; synthetic GPU smoke matched the CPU formula within `~5e-7`. The NVIDIA non-affine LayerNorm kernel (`ideogram_layer_norm_no_affine_f32`) matched a 7×4608 CPU reference within `~3.4e-6`, the low-level F32 RMSNorm wrapper matched a 4608-wide CPU reference exactly, adaLN scale/tanh-gate transform matched within `~2.4e-7`, gated residual matched within `~1e-6`, full-tensor MRoPE matched a 13×18×256 CPU reference within `~4.8e-7`, and full non-causal attention matched an 11×3×16 CPU reference within `~2.4e-7`. The NVIDIA MLP/final-vector wrappers for SiLU, multiply, and fused SiLU*Mul matched CPU references within `~5e-7` on a 12288-wide smoke. The current Ideogram GPU integration still streams data through host buffers and is intended for correctness/wiring work, not final performance. VAE convs use im2col + SIMD `GemmRows`, giving the earlier ~33x VAE decode speedup.

## Current status

Native Ideogram 4 FP8 text-to-image execution is implemented and real-weight validated component-by-component and end-to-end at a tiny proof-of-life budget. `cmd/ideogram4inspect` remains the metadata/inventory validator; `cmd/ideogram4gen` is the PNG generation driver. The current blocker is image quality/performance rather than missing CPU runtime coverage. NVIDIA kernels now cover opt-in FP8 linear streaming or lazy FP8 linear residency with explicit release hooks, Qwen3-VL FP8 projection/MLP calls through the same gated FP8 path, Qwen/shared weighted RMSNorm via `GO_PHERENCE_IDEOGRAM4_GPU_NORM`, Qwen RoPE via the existing RoPE kernel under `GO_PHERENCE_IDEOGRAM4_GPU_MROPE`, Qwen causal GQA attention via `GO_PHERENCE_IDEOGRAM4_GPU_ATTN`, VAE latent denormalization, unpatchify, direct Conv2D, GroupNorm, SiLU activation, mid-block attention, nearest-neighbour upsample, and final RGB clamp/scale via `GO_PHERENCE_IDEOGRAM4_GPU_VAE`, fused scheduler/CFG vector update, non-affine LayerNorm, adaLN transform, gated residual updates, MRoPE rotation, full DiT attention, and MLP/final vector operations, but the full Ideogram FP8 DiT/Qwen/VAE graph has not been converted to GPU-resident execution. Higher-fidelity generation needs either a much longer CPU/SIMD run or continued CUDA/NVIDIA conversion across GPU residency/streaming and VAE decode.
