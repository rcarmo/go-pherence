# Supported models and formats

This page tracks model architecture and weight-format support. The top-level README stays high-level; detailed backend and validation state lives under `docs/`.

## Performance snapshot

| Model | Arch | Format | GPU tok/s | CPU tok/s |
|---|---|---|---|---|
| **Qwen2.5-7B** | qwen2 | MLX 4-bit | **~120–158** | 1.1 |
| **SmolLM2-135M** | llama | BF16 | **86** | 35.5 |
| **Gemma3-1B** | gemma3 | MLX 4-bit | **~72** | 4.9 |
| **Qwen2.5-7B** | qwen2 | GPTQ 4-bit | **51** | 0.9 |
| **Qwen2.5-0.5B** | qwen2 | MLX 4-bit | **31** | 7.2 |
| **Qwen3-0.6B** | qwen3 | MLX 4-bit | **25** | 7.2 |
| **Gemma4-E2B** | gemma4 | MLX 4-bit | **~21–22** | — |
| **Qwen3-30B MoE** | qwen3_moe | MLX 4-bit | **~5.2 cold / ~5.5 warm** | 0.6 |

RTX 3060 12GB + i7-12700 6-core. Pure Go, zero CGo. Short-run decode rates vary with prompt length, route-set warmth, and VRAM headroom.

Qwen3 MoE uses native router softmax/top-k and selected-expert execution; optional REAP metadata can mask pruned experts before routing. The GGUF path understands Qwen3Next hybrid attention/SSM metadata, keeps supported quantised matrices packed, and can attach TurboQuant caches only to layers that own autoregressive K/V.

Use `make gguf-ci-qwen36-reap` for the pinned local Qwen3.6 REAP/TurboQuant gate. It checks metadata, tensor/layout readiness, one-token generation, SIMD rotation, KV/scratch accounting and benchmark counters. The detailed checkpoint investigation lives in [Qwen3.6 MTP](qwen36-mtp.md); TurboQuant policy and byte accounting live in [TurboQuant](turboquant.md).

## Architecture support

| Architecture | Models | Formats | Status |
|---|---|---|---|
| **llama** | SmolLM2, LLaMA 3.x | BF16, F16, F32 | ✅ |
| **qwen2** | Qwen2.5 0.5B–7B | MLX 4-bit, GPTQ 4-bit | ✅ |
| **qwen3** | Qwen3 0.6B+ | MLX 4-bit, BF16 | ✅ |
| **qwen3_moe** | Qwen3-30B-A3B MoE, Qwen3.6 REAP-pruned MoE route masks | MLX 4-bit; GGUF Q4_K/Q6_K REAP checkpoints | ✅ pure Go/SIMD MoE routing + optional static REAP expert masks; native GGUF inspect/smoke/bench/validation via `cmd/models/ggufinspect`, `cmd/models/ggufsmoke`, and `make gguf-ci-qwen36-reap`, including required SIMD rotation readiness plus TurboQuant KV/scratch/aggregate assertions |
| **gemma3** | Gemma 3 1B+ | MLX 4-bit, BF16 | ✅ |
| **gemma4** | Gemma 4 E2B+ | MLX 4-bit | ✅ |
| **lfm2_moe** | LFM2.5-8B-A1B hybrid conv/attention MoE | HF safetensors BF16 | 🧭 metadata/inspect coverage; runtime not implemented |
| **qwen3_tts** | Qwen3-TTS 0.6B/1.7B speech synthesis | HF safetensors + Qwen tokenizer + speech tokenizer | 🧭 metadata/token/inspect coverage; runtime not implemented |
| **moss_transcribe_diarize** | OpenMOSS MOSS-Transcribe-Diarize (Whisper-Medium encoder + VQ adaptor + Qwen3 decoder) | HF safetensors BF16 + tokenizer/processor sidecars | ✅ native greedy inference from 16 kHz mono PCM WAV; automatic zero-CGo NVIDIA PTX with resident Whisper/adaptor/Qwen3 execution and transparent CPU/SIMD fallback; recording-local speaker labels and text/raw/JSON/SRT/ASS exports; sampling, arbitrary media decode, and stable cross-recording speaker identity are not supported |
| **minicpmv / minicpmo** | OpenBMB MiniCPM-V and MiniCPM-O vision/omni chat checkpoints | HF safetensors + MiniCPM/Qwen tokenizer + processor/generation sidecars | 🧭 config, processor/tokenizer/generation metadata, image preprocessing, prompt/special-token planning, tensor inventory/shape validation, text/vision/resampler/audio execution planning, and `minicpmvinspect`; full tensor execution pending |
| **hunyuan3d_dit** | Hunyuan3D-2/2mini/2mv shape generation | HF safetensors + YAML | 🧪 native Go loader/runtime scaffold, image preprocessing, FlowMatch scheduling, tensor binding, fixture comparators, and CPU/SIMD ViT primitive layer; end-to-end generation pending conditioner tensor binding, DiT, ShapeVAE, and mesh kernels |
| **zimage_turbo** | Z-Image-Turbo text-to-image generation | Diffusers safetensors + Qwen tokenizer + AutoencoderKL | 🧭 metadata/inspect coverage; runtime not implemented |
| **diffusion_gemma** | DiffusionGemma 26B-A4B-it block-diffusion text generation | HF safetensors + Gemma tokenizer/processor | 🧪 native sparse text path: full 30-layer CPU/SIMD text stack, real weights, sparse top-k LM head, validated up to published 256-token canvas with 1/2 denoising steps; reference-complete parity and processor/vision integration pending |
| **ideogram4** | Ideogram 4 FP8 text-to-image generation | Diffusers safetensors + Qwen3-VL text encoder + AutoencoderKLFlux2 | ✅ native CPU/SIMD runtime via `cmd/image/ideogram4gen`; real-weight component and tiny end-to-end proof-of-life validation; no CUDA/NVIDIA Ideogram path yet |

Speech support has two separate native paths: the Whisper commands provide transcription/translation and optional ECAPA diarization, while `cmd/audio/moss-transcribe` runs the pinned MOSS speech encoder, adaptor, and Qwen3 decoder end to end with recording-local speaker labels and timestamps. See [moss-transcribe-diarize.md](moss-transcribe-diarize.md).

Any model from [mlx-community](https://huggingface.co/mlx-community) using the supported LLM architectures should work through the compatible loader path. LFM2.5-8B-A1B is a separate hybrid convolution/full-attention MoE decoder family and is tracked in [lfm2-moe-support.md](lfm2-moe-support.md). Qwen3-TTS is a separate multi-stage speech pipeline (talker, code predictor, codec decoder, optional speaker/codec encoder) and is tracked in [qwen3-tts-support.md](qwen3-tts-support.md). MiniCPM-V/O is tracked in [minicpmv-support.md](minicpmv-support.md): metadata/inspection and planning are covered by `make minicpmv-check`, while full vision/resampler/audio/text execution is still pending. Hunyuan3D-2 is a separate diffusion/3D pipeline family rather than an LLM decode architecture; native Go loading, config/tensor validation, image preprocessing, FlowMatch scheduling, fixture comparison, and initial CPU/SIMD ViT primitives exist, but full image-to-shape generation remains pending conditioner weight binding plus DiT/ShapeVAE/mesh kernels. See [hunyuan3d-2-support.md](hunyuan3d-2-support.md). Z-Image-Turbo is a separate text-to-image S3-DiT pipeline; config inspection is tracked in [zimage-turbo-support.md](zimage-turbo-support.md), with native Go/SIMD AVX/NEON/RVV implementation required before generation is marked ready. Ideogram 4 FP8 is another separate text-to-image pipeline with Qwen3-VL hidden-state conditioning, conditional/unconditional Ideogram4 DiTs, FlowMatch Euler scheduling, and AutoencoderKLFlux2 decode; native CPU/SIMD generation and validation status are tracked in [ideogram4-support.md](ideogram4-support.md).

## Weight format support

| Format | Detection | Dequant | GPU | Notes |
|---|---|---|---|---|
| **MLX affine 4-bit** | `config.json` quantization block | `val × scale + bias` | Transpose → GPTQ/MLX kernels | Primary format; runtime validates packed shape and F32/F16/BF16 scale/bias dtypes. |
| **GPTQ INT4** | `quantize_config.json` | `(val - 8) × scale` | Native tiled GEMV | Symmetric; runtime validates qweight/g_idx/scales/qzeros and Q4 GEMV inputs. |
| **BF16** | safetensors dtype | Direct load | F32/GPU BF16 helpers | Half bandwidth. |
| **F16** | safetensors dtype | F16→F32 at load | F32 on GPU | Compatibility path. |
| **F32** | safetensors dtype | Direct load | Native | Reference path. |
| **NVFP4 / FP4** | `quantization_config` ModelOpt/compressed-tensors metadata | FP4 E2M1 + F8_E4M3FN scale reference path | Upload + dequant-to-F32 fallback kernel | Experimental/internal only; synthetic CPU/NVIDIA dequant agrees, but public loading rejects NVFP4 until real checkpoint logits/tokens agree. |

## LFM2.5-8B-A1B roadmap

`LiquidAI/LFM2.5-8B-A1B` is tracked as a future architecture family rather than as a supported Qwen/LLaMA-compatible checkpoint.

Initial config inventory:

```text
model_type: lfm2_moe
architecture: Lfm2MoeForCausalLM
hidden_size: 2048
layers: 24
attention heads: 32 query / 8 KV
experts: 32 total / 4 active per token
layer pattern: conv + full_attention hybrid
conv_L_cache: 3
format: BF16 safetensors
```

See [lfm2-moe-support.md](lfm2-moe-support.md). Metadata/config inspection (`model/lfm2/config.go` and `cmd/models/lfm2inspect`) is in place; next steps are reference fixtures and CPU block parity, not generation shortcuts.

## Qwen3.6 MTP candidates

Current recommended Qwen native-MTP stress target:

```text
samwang0041/Qwen3.6-27B-MLX-4bit-MTP
```

Summary:

```text
model_type: qwen3_5
hidden_size: 5120
num_hidden_layers: 64
mtp_num_hidden_layers: 1
format: MLX affine 4-bit, group_size=64
size: ~15.37GB safetensors
```

Alternative similar candidate: `kradih/Qwen3.6-27B-MTP-4bit-MLX`. Larger MoE candidates exist, but they are less attractive for the current loader/runtime work. See [qwen36-mtp.md](qwen36-mtp.md).

## Gemma4 MTP model pairs

Use [mtp-speculative.md](mtp-speculative.md) and [gemma4-31b-runbook.md](gemma4-31b-runbook.md) for current MTP status.

Recommended local development pair:

```text
models/gemma4-e4b-it-4bit
models/gemma4-e4b-mtp-drafter
```

This E4B pair fully fits on the local RTX 3060 with current loader behavior (`42/42` layers resident, compact MLX LM head resident, ~5GiB VRAM free) and is much faster than the 31B stress path.
