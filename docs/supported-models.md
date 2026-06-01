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

MoE note: 128 experts/layer, 8 active/token. The pure Go/SIMD path runs router softmax/top-k and selected MLX 4-bit experts without llama.cpp. Optional REAP metadata (`reap_config.json` or `reap.json`) can statically mask pruned experts before top-k selection. TurboQuant-compatible KV policy names (`turbo4`, `turbo2`, `q8_0`, `f16`) are accepted by `cmd/llmserver` and `cmd/ggufsmoke`, and mapped to the native Go compressed-KV cache rather than external llama.cpp storage. `cmd/ggufinspect` provides lightweight GGUF tensor/metadata readiness checks for Q4_K/Q4_K_M-style local files before full runtime wiring. The GGUF config parser now recognizes architecture-scoped Qwen/Qwen-MoE metadata (`general.architecture`, `*.expert_count`, `*.expert_used_count`, `*.expert_feed_forward_length`) and fails explicitly for MoE runtime execution until the pure Go GGUF MoE forward path is wired. GGUF Q4_K row dequantization and `[inDim,outDim,experts]` expert tensor slicing are now available for `ffn_gate_exps`, `ffn_up_exps`, and `ffn_down_exps` wiring; the GGUF forward path now has a pure Go router → REAP mask → top-k → selected expert gate/up/down GEMV path for MoE layers. `ggufinspect -require-runtime-ready` now distinguishes tensor-index/quant readiness from actual generation readiness; the local `qwen35moe` REAP checkpoint is detected as Q4_K/Q6_K MoE-ready. The GGUF parser now captures Qwen3Next hybrid metadata (`attention.key_length`, `attention.value_length`, `full_attention_interval`, `ssm.*`) and loads fused `attn_qkv`/SSM tensor handles. Fused `attn_qkv` projection/splitting now matches the local REAP dimensions (`key=2048`, `value=4096`, `conv=8192`); QwenNext recurrent state sizing, depthwise conv state/update helpers, SSM delta update, gated value-head RMSNorm helpers, and `GGUFLlama.ForwardState` hybrid block integration are in place; the local Qwen3.6 REAP GGUF now infers REAP20 metadata from filename/model name, load-smokes with `reap=0.20`, plans native TurboQuant KV as `turbo4/turbo2` (`kv_dim=256`, 40 layers), and executes a one-token pure Go/SIMD forward smoke with quantized embeddings/LM head kept packed, producing the expected 248,320-vocab logits. NVIDIA backend runs attention, router, and selected experts via a GPU-resident expert cache; cold route sets pay one-time expert upload cost.

## Architecture support

| Architecture | Models | Formats | Status |
|---|---|---|---|
| **llama** | SmolLM2, LLaMA 3.x | BF16, F16, F32 | ✅ |
| **qwen2** | Qwen2.5 0.5B–7B | MLX 4-bit, GPTQ 4-bit | ✅ |
| **qwen3** | Qwen3 0.6B+ | MLX 4-bit, BF16 | ✅ |
| **qwen3_moe** | Qwen3-30B-A3B MoE, Qwen3.6 REAP-pruned MoE route masks | MLX 4-bit; GGUF Q4_K metadata/readiness inspection | ✅ pure Go/SIMD MoE routing + optional static REAP expert masks; GGUF inspection via `cmd/ggufinspect` |
| **gemma3** | Gemma 3 1B+ | MLX 4-bit, BF16 | ✅ |
| **gemma4** | Gemma 4 E2B+ | MLX 4-bit | ✅ |
| **lfm2_moe** | LFM2.5-8B-A1B hybrid conv/attention MoE | HF safetensors BF16 | 🧭 metadata/inspect coverage; runtime not implemented |
| **qwen3_tts** | Qwen3-TTS 0.6B/1.7B speech synthesis | HF safetensors + Qwen tokenizer + speech tokenizer | 🧭 metadata/token/inspect coverage; runtime not implemented |
| **hunyuan3d_dit** | Hunyuan3D-2/2mini/2mv shape generation | HF safetensors + YAML | 🧭 assessed, not implemented |

Any model from [mlx-community](https://huggingface.co/mlx-community) using the supported LLM architectures should work through the compatible loader path. LFM2.5-8B-A1B is a separate hybrid convolution/full-attention MoE decoder family and is tracked in [lfm2-moe-support.md](lfm2-moe-support.md). Qwen3-TTS is a separate multi-stage speech pipeline (talker, code predictor, codec decoder, optional speaker/codec encoder) and is tracked in [qwen3-tts-support.md](qwen3-tts-support.md). Hunyuan3D-2 is a separate diffusion/3D pipeline family rather than an LLM decode architecture; config/tensor inventory scaffolding exists, but runtime support is not implemented. See [hunyuan3d-2-support.md](hunyuan3d-2-support.md).

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

See [lfm2-moe-support.md](lfm2-moe-support.md). Metadata/config inspection (`model/lfm2/config.go` and `cmd/lfm2inspect`) is in place; next steps are reference fixtures and CPU block parity, not generation shortcuts.

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
