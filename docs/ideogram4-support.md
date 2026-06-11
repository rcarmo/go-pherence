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

1. **Inspection/readiness** — `loader/config/ideogram4.go`, `model/ideogram4` shape helpers, and `cmd/image/ideogram4inspect` validate local Diffusers folders, transformer inventories, FP8 scale coverage, and linear-role coverage.
2. **Qwen3-VL text conditioning** — `QwenVLConditioner` runs the text-only Qwen3-VL stack and concatenates the 13 selected hidden states into `[tokens, 53248]` conditioning features.
3. **Ideogram4 DiT forward** — native single-stream text+image token transformer blocks implement QK-RMSNorm, MRoPE, SwiGLU MLP, AdaLN timestep conditioning, final projection, and velocity prediction.
4. **FP8 weight loading and GEMV** — conditional and unconditional DiT weights plus Qwen linears load directly from `F8_E4M3` safetensors. GEMV dequantizes on the fly through the fp8 backend, with an amd64 AVX2/FMA gather-dot kernel and scalar fallback.
5. **Unconditional transformer / asymmetric CFG** — `DenoiseLoop` runs conditional and unconditional DiT paths and blends them with `CombineCFG`.
6. **FlowMatch Euler scheduler** — the native logit-normal schedule and Euler latent update match the reference scheduler numerically for inspected step plans.
7. **AutoencoderKLFlux2 decode** — `VAEDecoder` decodes denoised 32-channel latents to RGB output.
8. **End-to-end CLI** — `cmd/image/ideogram4gen` loads tokenizer, Qwen3-VL, both DiTs, scheduler, and VAE from a Diffusers folder and writes a PNG.

The project has a separate NVIDIA backend for LLM/Gemma/Qwen paths. Ideogram 4 now has CUDA/NVIDIA primitives for fused scheduler/CFG vector updates, weighted RMSNorm, row-wise non-affine LayerNorm, adaLN scale/gate transforms, gated residual updates, full-tensor MRoPE/RoPE rotation, full attention (DiT non-causal and Qwen/VAE variants), MLP/final vector operations (SiLU, multiply, fused SiLU*Mul), VAE latent denorm/unpatchify/direct Conv2D/GroupNorm/upsample/RGB clamp, and an experimental FP8 E4M3 linear GEMV. `cmd/image/ideogram4gen -gpu` now enables only the coarse production-safe GPU gates (CFG and VAE) and intentionally leaves token/row-level gates plus FP8 projection offload disabled because the current FP8, norm, RoPE/attention, and MLP paths are correctness-oriented fine-grained calls that are slower than CPU/SIMD for real renders unless batched execution is added. Use `-gpu-fp8` or `GO_PHERENCE_IDEOGRAM4_GPU_FP8=1` for FP8 diagnostics and the new DiT layer-scoped residency path: each DiT block uploads QKV/O/W1/W2/W3/AdaLN once, processes all tokens in that layer, then frees those weights before the next layer. `-gpu-fp8-cache`/`GO_PHERENCE_IDEOGRAM4_GPU_FP8_CACHE=1` keeps non-layer/global FP8 weights resident but can exceed 12GB VRAM if used too broadly. `GO_PHERENCE_IDEOGRAM4_GPU_RESIDENCY` controls coarse VRAM policy: `persistent` keeps cached weights until `ReleaseGPU`, `phase` releases cached DiT weights between Qwen/DiT/VAE phases for 12GB-class cards, and `stream` disables long-lived FP8 caching. `cmd/image/ideogram4gen` exposes `-gpu`, `-gpu-strict`, `-gpu-fp8`, `-gpu-fp8-cache`, and `-gpu-residency persistent|phase|stream`. `FP8Linear`, `DiTLayer`, `DiTModel`, and `NativePipeline` expose `ReleaseGPU` hooks, and `cmd/image/ideogram4gen` defers `pipe.ReleaseGPU()` to clean cached linears on exit. The individual env gates remain available: `GO_PHERENCE_IDEOGRAM4_GPU_CFG`, `_NORM`, `_MROPE`, `_ATTN`, `_MLP`, and `_VAE`, each with a `_STRICT` counterpart.

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

**Validated against real weights:** `cmd/image/ideogram4vaesmoke` loads the actual downloaded `vae/diffusion_pytorch_model.safetensors` (251 tensors) and decodes a random `[32,16,16]` latent end-to-end through the full conv graph, producing a valid `128x128` RGB image with a healthy pixel distribution (e.g. min=27 max=188 mean=111). All tensor names/shapes matched on the first try. `Conv2D` uses im2col + the SIMD `GemmRows` kernel, cutting the `128x128` decode from ~3m25s to ~6s (~33x) with bit-identical output.

## Qwen3-VL text conditioning

`model/ideogram4/qwen_vl_conditioner.go` (`QwenVLConditioner`) is the native Qwen3-VL text-only forward (`hidden=4096`, 36 layers, 32 q / 8 kv GQA heads, `head_dim=128`, SwiGLU `intermediate=12288`, RMSNorm `1e-6`). Linears are the checkpoint's FP8 E4M3 weights (loaded via the fp8 backend); embeddings are decoded per-token from bf16 to avoid materializing the full table. Each layer: RMSNorm → q/k/v FP8 proj → per-head q/k RMSNorm → RoPE → causal GQA attention → o-proj residual → RMSNorm → SwiGLU MLP residual. `Condition(tokenIDs)` captures hidden states at `ActivationLayers` (`[0,3,...,35]`, HF indexing with 0=post-embedding) and concatenates them into the `[tokens, 53248]` tensor feeding `DenoiseLoop`.

## End-to-end native pipeline

`model/ideogram4/native_pipeline.go` (`NativePipeline` / `LoadNativePipeline`) assembles every component from a Diffusers directory — tokenizer, Qwen3-VL conditioner, conditional + unconditional FP8 DiT transformers, and the VAE decoder (single-file or sharded safetensors auto-detected). `NativePipeline.Generate(prompt, opts)` runs the full path: tokenize → Qwen3-VL conditioning → FlowMatch denoise loop with asymmetric CFG → unpatchify → VAE decode → RGB `Image`. `cmd/image/ideogram4gen` drives it from the CLI (seeded Gaussian init latents, PNG output).

The full text→image path is now implemented natively in Go/SIMD. The DiT, MRoPE, adaLN, final layer, latent denormalization/unpatchify, and conditioning were reconciled against the reference `ideogram-oss/ideogram4` source (`modeling_ideogram4.py`, `scheduler.py`, `latent_norm.py`, `pipeline_ideogram4.py`), resolving the earlier provisional assumptions:

- adaLN is **scale + tanh-gate** (not shift/scale): per block `4*emb` → `scale_msa, gate_msa, scale_mlp, gate_mlp`; sublayers are `x += tanh(gate) * norm2(sublayer(norm1(x)*(1+scale)))` with four learnable RMSNorms per block.
- attention uses **QK-RMSNorm** (`norm_q`/`norm_k`, eps 1e-5) before RoPE.
- **MRoPE** uses interleaved 3-axis section assignment over the full head_dim with `IMAGE_POSITION_OFFSET=65536` and text positions `(i,i,i)`.
- the final layer is a **non-affine LayerNorm** with scale-only modulation (`final_adaln` → `emb`, not `2*emb`).
- all bias=true linears load their `.bias`; `embed_image_indicator` and `llm_cond_norm` are applied; latents are denormalized with the per-channel `LATENT_SCALE/SHIFT` constants and unpatchified in `(patch_h, patch_w, ae_channels)` order before VAE decode.

The prompt is wrapped in the Qwen3-VL ChatML template (`TokenizeChatPrompt`). The byte-level BPE encoder now matches the reference HuggingFace tokenizer on the local 15-prompt validation set, including emoji, CJK, code, URLs, tabs, punctuation, and mixed whitespace.

## Validation status against real weights

The implementation has been validated on real gated weights as follows:

- **VAE decoder** — fully downloaded (168 MB) and run end-to-end (`cmd/image/ideogram4vaesmoke`): all 251 tensors load, the conv graph executes, and a `[32,16,16]` latent decodes to a valid `128x128` RGB image. im2col + SIMD `GemmRows` makes it ~6s. The decode graph was also reconciled against the reference `autoencoder.py`/`_decode` (mid `block_1/attn_1/block_2`, up-blocks 0–3 with 3 resnets each and upsample after 0–2, `AttnBlock`/`ResnetBlock`/nearest-`Upsample` semantics, 128-dim latent denorm before unpatch, `(x+1)*127.5` output) — no discrepancies found.
- **FP8 DiT transformer** — the safetensors header (all 669 tensor shapes/dtypes) was fetched via a ~70 KB HTTP range request and checked against the loader's expectations: **0 shape mismatches, 0 missing tensors**. Weights are `F8_E4M3`, per-output-row scales are `F32`, and biases/norms are `BF16` (all handled by `decodeScale`/`decodeFloatVec`). This confirms the corrected layout (e.g. `final_layer.adaln_modulation` → `emb`, `embed_image_indicator` `[2,emb]`, QK-norms `[head_dim]`, four per-block RMSNorms, all biases). The **full 34-layer forward was also executed on the real weights** (transformer downloaded transiently): `Velocity` over a tiny token set ran with **0 NaN/Inf** and numerically healthy output (mean≈0, std≈0.98 — as expected for a flow-matching velocity), confirming the DiT graph is correct end-to-end, not just shape-consistent.
- **Qwen3-VL text encoder** — same header-only validation over all 1117 tensors: **0 errors**. FP8 q/k/v/o and gate/up/down projections with `F32` per-row scales, `BF16` q/k/input/post norms, and a `BF16` `[vocab, hidden]` embedding table — exactly what `QwenVLConditioner` loads. The **full 36-layer forward was also executed on the real weights** (encoder downloaded transiently): `Condition` produced the `[tokens, 53248]` features with **0 NaN/Inf**; magnitudes are large (std≈63) as expected for raw concatenated LLM hidden states with outlier activations, which the DiT's `llm_cond_norm` RMSNorm then normalizes.

- **Schedule** — the FlowMatch logit-normal schedule was checked numerically against the reference `scheduler.py`: for a 6-step plan the t-value sequence matches (`[0.007, 0.528, 0.657, 0.746, 0.819, 0.886]`) to the clamp constant.
- **Tokenizer** — the byte-level BPE `Encode` path was validated against the reference HuggingFace `tokenizers` library on 15 diverse prompts (emoji, CJK, code, URLs, tabs, mixed whitespace, punctuation): **15/15 exact token-ID matches**, using the Qwen2/Qwen3 pre-tokenization regex plus a `splitWhitespaceRuns` emulation of the RE2-inexpressible `\s+(?!\S)` lookahead.

### End-to-end run on real weights (staged)

An early **conditional-only** (CFG-disabled) staged generation was run under disk pressure by downloading one large component at a time. The full path executed on real weights end-to-end (12 prompt tokens encoded; 6 DiT forwards; VAE decode → `64x64` RGB PNG) with finite values throughout and no crashes. The output was a low-detail wash, which was expected for the deliberately degenerate budget (64×64, 6 steps, no classifier-free guidance).

After freeing workspace scratch, the full Diffusers folder (text encoder, conditional DiT, unconditional DiT, VAE, tokenizer, configs; roughly 27 GB) was downloaded under `/workspace/tmp` and a CFG-guided cat prompt was run through `cmd/image/ideogram4gen` at `64x64`, 6 steps, guidance `5.0`, seed `42`. The optimized CPU/SIMD path completed in about `529s` and produced a finite PNG. The image was still not a recognizable cat; it should be treated as an execution proof-of-life, not as a quality benchmark. A faithful sample likely needs higher resolution and substantially more steps, which is currently expensive on the CPU-only Ideogram path.

**Performance:** the CPU FP8 E4M3 decode uses a 256-entry lookup table (one byte → one of 256 values), making the dequant GEMV branch-free. The amd64 backend adds an AVX2/FMA `VGATHERDPS` E4M3 dot kernel over that LUT; a standalone `4096x4608` FP8 GEMV smoke measured about `3.04ms` versus `8.85ms` for the scalar LUT loop (`~2.9x` speedup, with expected accumulation-order drift around `1e-3`). The NVIDIA runtime also has a direct FP8 E4M3 GEMV kernel (`fp8_e4m3_gemv_f32`) and `GPUFP8E4M3Linear` upload wrapper; synthetic GPU smoke matched the CPU backend within `~1e-5`, and the cached `FP8Linear.Apply` path matched repeated CPU calls within `~2e-5`. The fused NVIDIA Ideogram CFG+FlowMatch step (`ideogram_cfg_step_f32`) replaces `guided = uncond + guidance*(cond-uncond)` plus `x += sigma*guided`; synthetic GPU smoke matched the CPU formula within `~5e-7`. The NVIDIA non-affine LayerNorm kernel (`ideogram_layer_norm_no_affine_f32`) matched a 7×4608 CPU reference within `~3.4e-6`, the low-level F32 RMSNorm wrapper matched a 4608-wide CPU reference exactly, adaLN scale/tanh-gate transform matched within `~2.4e-7`, gated residual matched within `~1e-6`, full-tensor MRoPE matched a 13×18×256 CPU reference within `~4.8e-7`, and full non-causal attention matched an 11×3×16 CPU reference within `~2.4e-7`. The NVIDIA MLP/final-vector wrappers for SiLU, multiply, and fused SiLU*Mul matched CPU references within `~5e-7` on a 12288-wide smoke. The current Ideogram GPU integration still streams data through host buffers and is intended for correctness/wiring work, not final performance. VAE convs use im2col + SIMD `GemmRows`, giving the earlier ~33x VAE decode speedup.

## Current status

Native Ideogram 4 FP8 text-to-image execution is implemented and real-weight validated component-by-component and end-to-end at a tiny proof-of-life budget. `cmd/image/ideogram4inspect` remains the metadata/inventory validator; `cmd/image/ideogram4gen` is the PNG generation driver. The current blocker is image quality/performance rather than missing CPU runtime coverage. NVIDIA kernels now cover opt-in FP8 linear streaming or lazy FP8 linear residency with explicit release hooks, Qwen3-VL FP8 projection/MLP calls through the same gated FP8 path, Qwen/shared weighted RMSNorm via `GO_PHERENCE_IDEOGRAM4_GPU_NORM`, Qwen RoPE via the existing RoPE kernel under `GO_PHERENCE_IDEOGRAM4_GPU_MROPE`, Qwen causal GQA attention via `GO_PHERENCE_IDEOGRAM4_GPU_ATTN`, VAE latent denormalization, unpatchify, direct Conv2D, GroupNorm, SiLU activation, mid-block attention, nearest-neighbour upsample, and final RGB clamp/scale via `GO_PHERENCE_IDEOGRAM4_GPU_VAE`, fused scheduler/CFG vector update, non-affine LayerNorm, adaLN transform, gated residual updates, MRoPE rotation, full DiT attention, and MLP/final vector operations, but the full Ideogram FP8 DiT/Qwen/VAE graph has not been converted to GPU-resident execution. Higher-fidelity generation needs either a much longer CPU/SIMD run or continued CUDA/NVIDIA conversion across GPU residency/streaming and VAE decode.

## Current optimized NVIDIA profile (RTX 3060 12GB)

The current local performance profile is tuned for an RTX 3060 12GB with model files on SSD and uses structured JSON prompts matching the ComfyUI-Ideogram4 Magic Prompt → Generate workflow. Reproducible targets live in the top-level `Makefile`:

```bash
make ideogram4-cat-gpu-256
make ideogram4-cat-gpu-512
make ideogram4-residency-sweep
make ideogram4-vae-probe
```

The 256px default path enables:

```text
-gpu -gpu-fp8 -gpu-fp8-cache -gpu-fp8-sgemm -gpu-residency phase
GO_PHERENCE_IDEOGRAM4_GPU_DIT_VECTOR=1
GO_PHERENCE_IDEOGRAM4_GPU_FULL_LAYER=1
GO_PHERENCE_IDEOGRAM4_GPU_HIDDEN_RESIDENT=1
GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW_COND=34
GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW_UNCOND=9
```

This profile keeps the DiT hidden state resident across the layer loop, runs the attention and MLP sublayers as GPU-resident islands through post-norm and gated residual, keeps all conditional DiT layer weights resident, and keeps the first nine unconditional layers resident. AdaLN residency is available behind `GO_PHERENCE_IDEOGRAM4_GPU_ADALN_RESIDENT=1` but is default-off because it exceeds 12GB VRAM with the aggressive default cache profile.

Validated 256×256, 8-step structured cat prompt timing:

```text
qwen_condition ≈ 12.5s
DiT denoise    ≈ 9m25s
VAE decode     ≈ 4.6s
total generate ≈ 9m43s
```

The output is a recognizable orange tabby cat. The early correct baseline for the same 8-step prompt was roughly 16m22s, so the current profile is about 40% faster end-to-end.

### Residency and transfer findings

Instrumentation is enabled with:

```text
GO_PHERENCE_IDEOGRAM4_GPU_STATS=1
GO_PHERENCE_IDEOGRAM4_TIMING=1
```

The first instrumented denoise path had millions of tiny transfers/allocations per step. Batched DiT RMSNorm, batched post-norm/gated residual, reusable GPU scratch, and GPU-resident attention/MLP islands reduced 1-step denoise from roughly:

```text
initial: kernels≈957k, h2d≈1.96M, d2h≈957k, denoise≈1m45s
current: kernels≈2.6k, h2d≈3.3k, d2h≈1.3k, denoise≈1m12s
```

Hidden-resident full-layer mode reduced DiT D2H to a small tail. Current 8-step denoise still uploads tens of GB host→device, mostly streamed layer weights/dynamic inputs. The asymmetric cache profile is the best stable 12GB tradeoff found so far:

```text
COND cache=34, UNCOND cache=9: stable
UNCOND>=10 with COND=34: cuMemAlloc error 2
symmetric window=21: stable
symmetric window>=22: CUDA error 700 / allocation failure
```

Role-wide QKV/O/attention residency knobs exist for experiments:

```text
GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_QKV_ALL
GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_O_ALL
GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_ATTENTION_ALL
```

but are default-off because all-layer role residency exceeds VRAM in combination with the current hidden-resident profile.

### Resolution limits

At 512×512, the 256px residency defaults exceed VRAM. The conservative 512 target uses:

```text
COND cache=16
UNCOND cache=0
steps=4 by default
```

Measured 512×512, 4-step timing:

```text
qwen_condition ≈ 12.7s
DiT denoise    ≈ 15m58s
VAE decode     ≈ 54s
total generate ≈ 17m05s
```

The 512 output is recognizable but artifacted at 4 steps. 512px is currently the practical upper bound for routine use on this 12GB GPU. 768px is experimental and likely hour-scale for a 4-step render; 1024px is not practical without tiled/chunked attention.

### Remaining bottlenecks

The remaining high-value work is:

1. reduce DiT H2D further with safer selective weight/dynamic-buffer residency without exceeding 12GB,
2. implement real VAE feature-map residency using the buffer-level VAE primitives,
3. investigate tiled/chunked attention for 768px/1024px feasibility.

VAE GPU buffer primitives now exist for Conv2D, GroupNorm, UpsampleNearest, and RGB clamp, and `make ideogram4-vae-probe` provides a reproducible VAE-only benchmark. A 512px VAE probe currently shows ~53s decode with ~6.7GB H2D and ~6.2GB D2H, so high-resolution VAE needs full buffer residency/direct-conv improvements.

### Qwen/VAE optimization notes

`-gpu-fp8-sgemm` is enabled by default in the Makefile Ideogram targets because it consistently reduces Qwen conditioning time on the local RTX 3060 profile (roughly 17s → 12.5s for the structured cat prompt) without first-step semantic drift. It does not materially change current DiT denoise timing because the DiT path now mostly uses the hidden-resident full-layer islands rather than the older host-staged FP8 GEMM wrappers.

The VAE path has reusable scratch buffers and buffer-level primitives for Conv2D, GroupNorm, UpsampleNearest, RGB clamp, and the spatial attention building blocks. At 256px, VAE decode is a small tail (~4.5s). At 512px, the VAE becomes material (~53s): block timing shows the mid-block spatial attention dominates (`~42–43s` of the decode). A naive SGEMM-backed full attention experiment is available behind `GO_PHERENCE_IDEOGRAM4_GPU_VAE_ATTN_SGEMM=1`, but it was slower at 512px (`~45.6s` mid-attention versus `~42.8s`) and remains default-off. Higher-resolution VAE work should focus on tiled/streaming spatial attention rather than more direct-conv tuning.

### K3 A100 row-scale Q8 FP8 linears

`GO_PHERENCE_IDEOGRAM4_K3_A100_Q8=1` adds an opt-in A100/IME2 path for
`FP8Linear.ApplyBatch` on riscv64 K3 systems. When `-k3` (or
`GO_PHERENCE_IDEOGRAM4_K3=1`) is enabled and a linear has dimensions divisible by
32, the path converts the E4M3 FP8 weight to a resident row-scale `Q80x32` A100
layout, packs F32 activations on X100 worker goroutines, and dispatches the GEMM
through the documented A100 worker pool (`/proc/set_ai_thread`, cores 8+). Bias
is applied after the A100 GEMM. `-k3-prewarm` now pre-builds the Q8 row-scale
resident cache when this flag is enabled; otherwise it keeps the earlier FP16/RVV
prewarm behavior.

The path deliberately uses row-global Q8 weight scales, matching the
native-compatible row-scale strategy that stabilized Whisper's A100 FFN path. It
is still opt-in until validated against real Ideogram4 weights/images, but
synthetic E4M3 correctness tests compare it against the CPU FP8 reference within
Q8 tolerance.

Milk-V/K3 synthetic `ApplyBatch` benchmark (`go test ./model/ideogram4 -run '^$'
`-bench 'BenchmarkK3FP8ApplyBatch(DiTShape|RVVF16|A100Q8)' -benchtime=1x`):

| Shape | RVV/F16 K3 | A100 row-scale Q8 | Speedup |
|---|---:|---:|---:|
| batch=64, 512→1024 | 1.71 ms | 0.89 ms | 1.9× |
| batch=32, 4608→4608 | 54.7 ms | 7.14 ms | 7.7× |
| batch=16, 4608→12288 | 72.4 ms | 8.47 ms | 8.5× |

Use for focused experiments with:

```sh
GO_PHERENCE_IDEOGRAM4_K3=1 \
GO_PHERENCE_IDEOGRAM4_K3_A100_Q8=1 \
GO_PHERENCE_IDEOGRAM4_K3_A100_WORKERS=8 \
IME2_ACT_PACK_WORKERS=8 \
./ideogram4gen -k3 -k3-prewarm ...
```

### K3 fused A100 MLP W1/W3/W2

`GO_PHERENCE_IDEOGRAM4_K3_A100_MLP=1` adds an opt-in fused K3 MLP dispatch for
Ideogram DiT layers when the A100 row-scale Q8 path is also enabled. The fused
path packs the shared MLP input once and computes `W1(x)` and `W3(x)` in a single
A100 worker dispatch via `Gemm2Q80x32AIPooledX100PackSameInput`, applies the
existing K3/RVV `SiLU(W1) * W3` seam, then runs `W2` through the A100 row-scale
`FP8Linear.ApplyBatch` path. This keeps the numerical contract identical to the
unfused A100 row-scale path while reducing one activation pack and one A100 pool
dispatch for the W1/W3 pair.

Synthetic Milk-V/K3 DiT-like MLP benchmark (`batch=16`, `emb=4608`,
`intermediate=12288`, `GO_PHERENCE_IDEOGRAM4_K3_A100_Q8=1`):

| Path | Time | Notes |
|---|---:|---|
| Unfused A100 W1 + W3 + W2 | 38.91 ms | three independent ApplyBatch calls |
| Fused W1/W3 + A100 W2 | 38.45 ms | shared input pack/dispatch for W1+W3 |

The fused path is correctness-tested against the CPU FP8 reference on a synthetic
MLP. The current end-to-end MLP speedup is small because `W2` and the two large
A100 GEMMs dominate, but the helper is the right integration point for future
SiLU*Mul→Q8 activation packing and more aggressive W2 fusion.

### K3-only runtime policy and 128x128 smoke

On Milk-V/K3, `-k3` now hard-disables Ideogram NVIDIA paths: GPU enabled/strict
predicates return false, DiT GPU residency upload is skipped, and layer helpers
fall back to K3/X100/RVV/A100 or scalar CPU seams. There is intentionally no
NVIDIA escape hatch for K3 because this platform has no NVIDIA hardware.

Ideogram A100 defaults to all eight A100 cores (`8-15`) for FP8 row-scale Q8
linears. With the full `ideogram-ai/ideogram-4-fp8` snapshot downloaded,
`128x128`, one-step K3 smoke succeeds with:

```sh
GO_PHERENCE_IDEOGRAM4_K3=1 \
GO_PHERENCE_IDEOGRAM4_K3_A100_Q8=1 \
GO_PHERENCE_IDEOGRAM4_K3_A100_MLP=1 \
GO_PHERENCE_IDEOGRAM4_K3_A100_WORKERS=8 \
IME2_ACT_PACK_WORKERS=8 \
ideogram4gen -model /home/me/models/ideogram-4-fp8 \
  -prompt "a small red cube on a wooden table" \
  -out /tmp/ideogram4-k3-128.png -height 128 -width 128 \
  -steps 1 -seed 1 -k3 -timing
```

Measured on Milk-V/K3:

| Phase | Time |
|---|---:|
| load pipeline | 1.50 s |
| Qwen conditioning | 1m53.36s |
| denoise, 1 step | 5m14.66s |
| release denoise weights before VAE | 4.13s |
| VAE decode | 12.23s |
| total generation | 7m24.54s |
| PNG write | 16 ms |

The `release_denoise_before_vae` phase drops tokenizer/text/DiT references,
releases residency caches, and forces GC before VAE decode. Without this, the
same 128x128 run was killed after denoise/latent postprocess; 64x64 completed
without the release but the release is now the default in K3 mode.

### Selective K3 DiT prewarm

`-k3-prewarm` now warms only the DiT denoise FP8 linears by default instead of
all Qwen/text/auxiliary linears. The previous all-linears traversal built too
many A100 Q8 caches and could kill the process; it remains available for
experiments with `GO_PHERENCE_IDEOGRAM4_K3_PREWARM_ALL=1`.

With `GO_PHERENCE_IDEOGRAM4_K3_A100_Q8=1` and
`GO_PHERENCE_IDEOGRAM4_K3_A100_MLP=1`, selective prewarm builds 412 resident DiT
linear caches across conditional and unconditional transformers. It shifts the
first-use FP8→Q80 packing cost into `load_pipeline` and makes subsequent denoise
passes use already-resident A100 row-scale weights.

Milk-V/K3 timings:

| Run | Prewarm/load | Qwen condition | Denoise | VAE | Generate after load |
|---|---:|---:|---:|---:|---:|
| 64×64, 1 step | 7m10.99s | 2m44.77s | 13.22s | 3.27s | 3m04.64s |
| 128×128, 1 step | 7m12.02s | 2m46.71s | 31.51s | 12.34s | 3m34.09s |

For comparison, the non-prewarmed 128×128 one-step run spent about `8m02s` in
denoise and `11m00s` total generation after load. Selective prewarm is therefore
the right mode for repeated image generation or a resident service, while one-off
CLI use still pays the prewarm cost up front.

### Optional K3 Qwen conditioner prewarm

`GO_PHERENCE_IDEOGRAM4_K3_PREWARM_QWEN=1` extends `-k3-prewarm` to the Qwen3-VL
text conditioner linears. This builds and retains A100 row-scale Q8 caches for
the 36 text layers (Q/K/V/O plus gate/up/down projections) in addition to the
selective DiT denoise caches. It is opt-in because it increases prewarm time and
resident memory, but it is the right mode for a long-lived K3 service.

Milk-V/K3 64×64 one-step smoke with DiT+Qwen prewarm:

| Phase | Time |
|---|---:|
| prewarm/load | 9m53.83s |
| Qwen conditioner | 1.67s |
| denoise | 13.1s |
| VAE decode | 3.3s |

Without Qwen prewarm, the same prompt-conditioning phase was about `2m45s`; with
prewarm each Qwen layer is roughly `44–54ms` and the full conditioner is under
2 seconds. Use:

```sh
GO_PHERENCE_IDEOGRAM4_K3_PREWARM_QWEN=1 \
GO_PHERENCE_IDEOGRAM4_K3=1 \
GO_PHERENCE_IDEOGRAM4_K3_A100_Q8=1 \
GO_PHERENCE_IDEOGRAM4_K3_A100_MLP=1 \
ideogram4gen -k3 -k3-prewarm ...
```

### Expanded selective DiT global prewarm and steady-state profile

Selective K3 prewarm now also includes the DiT global linears used during denoise
(`llm_cond_proj`, `time_in`, `time_out`, `final_adaln`, `input_proj`, and
`final_linear`) in addition to per-layer AdaLN/QKV/O/W1/W2/W3 linears. With
`GO_PHERENCE_IDEOGRAM4_K3_PREWARM_QWEN=1`, the resident service profile now
builds 672 K3/A100 row-scale Q8 linears across Qwen and both DiT branches.

This removes the previous first-use global projection stall inside `Velocity`:
`llm_cond_proj` fell from about `6.94s` to about `0.12s` on a 64×64 conditional
branch, and the 64×64 one-step denoise phase fell from about `13.6s` to
`5.86s`.

Milk-V/K3 fully-prewarmed timings (`GO_PHERENCE_IDEOGRAM4_K3_A100_WORKERS=8`,
`IME2_ACT_PACK_WORKERS=8`):

| Run | Qwen | Denoise | VAE | Generate after load |
|---|---:|---:|---:|---:|
| 64×64, 1 step | 1.64s | 5.86s | 3.37s | 14.65s |
| 128×128, 1 step | 1.90s | 24.65s | 12.30s | 42.90s |

For 128×128, branch-level denoise timing after full prewarm is now dominated by
layer execution rather than global projections:

- conditional branch: `layers=14.24s`, globals/final ≈ `0.13s`
- unconditional branch: `layers=10.20s`, globals/final ≈ `0.07s`

The remaining K3 optimization target is therefore steady-state DiT layer kernels,
especially attention and MLP internals, not cache construction or scheduler/CFG.
