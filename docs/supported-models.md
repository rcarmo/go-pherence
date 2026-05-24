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

MoE note: 128 experts/layer, 8 active/token. NVIDIA backend runs attention, router, and selected experts via a GPU-resident expert cache; cold route sets pay one-time expert upload cost.

## Architecture support

| Architecture | Models | Formats | Status |
|---|---|---|---|
| **llama** | SmolLM2, LLaMA 3.x | BF16, F16, F32 | ✅ |
| **qwen2** | Qwen2.5 0.5B–7B | MLX 4-bit, GPTQ 4-bit | ✅ |
| **qwen3** | Qwen3 0.6B+ | MLX 4-bit, BF16 | ✅ |
| **qwen3_moe** | Qwen3-30B-A3B MoE | MLX 4-bit | ✅ |
| **gemma3** | Gemma 3 1B+ | MLX 4-bit, BF16 | ✅ |
| **gemma4** | Gemma 4 E2B+ | MLX 4-bit | ✅ |
| **hunyuan3d_dit** | Hunyuan3D-2/2mini/2mv shape generation | HF safetensors + YAML | 🧭 assessed, not implemented |

Any model from [mlx-community](https://huggingface.co/mlx-community) using the supported LLM architectures should work through the compatible loader path. Hunyuan3D-2 is a separate diffusion/3D pipeline family rather than an LLM decode architecture; see [hunyuan3d-2-support.md](hunyuan3d-2-support.md).

## Weight format support

| Format | Detection | Dequant | GPU | Notes |
|---|---|---|---|---|
| **MLX affine 4-bit** | `config.json` quantization block | `val × scale + bias` | Transpose → GPTQ/MLX kernels | Primary format; runtime validates packed shape and F32/F16/BF16 scale/bias dtypes. |
| **GPTQ INT4** | `quantize_config.json` | `(val - 8) × scale` | Native tiled GEMV | Symmetric; runtime validates qweight/g_idx/scales/qzeros and Q4 GEMV inputs. |
| **BF16** | safetensors dtype | Direct load | F32/GPU BF16 helpers | Half bandwidth. |
| **F16** | safetensors dtype | F16→F32 at load | F32 on GPU | Compatibility path. |
| **F32** | safetensors dtype | Direct load | Native | Reference path. |
| **NVFP4 / FP4** | `quantization_config` ModelOpt/compressed-tensors metadata | FP4 E2M1 + F8_E4M3FN scale reference path | Upload + dequant-to-F32 fallback kernel | Experimental/internal only; synthetic CPU/NVIDIA dequant agrees, but public loading rejects NVFP4 until real checkpoint logits/tokens agree. |

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
