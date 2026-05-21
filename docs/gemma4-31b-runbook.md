# Gemma 4 31B local runbook

## Local assets

Main 4-bit MLX checkpoint:

```text
models/gemma4-31b-it-4bit
```

MTP assistant/drafter 4-bit checkpoint:

```text
models/gemma4-31b-it-mtp-assistant-4bit
```

The main checkpoint is `mlx-community/gemma-4-31b-it-4bit` and uses sharded MLX safetensors. The assistant checkpoint is `guardiangate1775/gemma-4-31B-it-assistant-4bit` and is tagged as `speculative-decoding`/`mtp`/`drafter`.

## Hardware fit

Current container hardware:

- CPU: Intel i7-12700 exposed as 6 CPUs
- RAM: 62 GiB total, about 49 GiB available before loading
- GPU: RTX 3060 12 GiB, about 11.5 GiB free at idle

The 31B main model is about 18 GiB on disk in 4-bit MLX form. CPU dequant-at-load is not viable on this host; a plain CPU load was OOM-killed. Use on-the-fly quantized loading, currently enabled by the `llmgen -gpu` path.

## Current smoke status

This loads successfully:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/llmgen \
  -gpu -gpu-layers 1 \
  -model models/gemma4-31b-it-4bit \
  -tokens 0 \
  -prompt "Hello"
```

A plain CPU load is not recommended:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/llmgen \
  -model models/gemma4-31b-it-4bit \
  -tokens 0 \
  -prompt "Hello"
```

It attempts dequant-at-load and was killed by the OS.

## Loader notes

The 31B checkpoint has Gemma4-specific layout differences that should live in a dedicated Gemma4 loader path rather than being scattered through generic LLaMA loading:

- top-level `model_type=gemma4` with nested `text_config.model_type=gemma4_text`
- tensor names under `language_model.model.*`
- full-attention layers use `num_global_key_value_heads=4` and `global_head_dim=512`
- sliding-attention layers use `num_key_value_heads=16` and `head_dim=256`
- `attention_k_eq_v=true`; full-attention layers omit `v_proj` and must use/copy K as V

Initial normalization, per-layer K/V head selection, and K=V support are in place. `model/gemma` now owns Gemma4 nested config normalization and KV-head selection; the next cleanup should move K=V projection and tensor-prefix rules into that explicit loader boundary.

## Upload-path audit

The first 31B GPU smoke exposed the real bottleneck: the loader kept the tied embedding LM head only as dequantized F32, so `LoadGPUModelWithLayers` uploaded a 5.6 GiB F32 LM head and left too little VRAM for transformer layers. Retaining the packed MLX embedding as `LMHeadMLX` for tied LM-head checkpoints enables the compact MLX LM-head upload path:

```text
[model] trying compact MLX LM head (packed 705 MB, f32 5637 MB, free 7581 MB, in=5376 out=262144 group=64)
[model] MLX LM head on GPU (packed 705 MB, f32 5637 MB)
[budget] GPU VRAM: 5520/11910 MB used (6389 MB free)
```

So the immediate effective run strategy is mmap/on-the-fly MLX loading plus compact MLX LM head, then incrementally increase `-gpu-layers` until VRAM is saturated. The remaining ~250s wall time for `-tokens 0` is prompt processing/chat-template forward work, not mmap load time; model load is ~15–16s and one-layer GPU upload is ~2.5s after compact LM head.

## MTP deployment audit

MTP is not currently deployed through `cmd/llmgen`. The `-speculative` flag exposes only the stock-weight proposer/replay verifier path; it does not load `Gemma4MTPDrafter` or the local assistant weights.

Current Gemma4 MTP status:

- `LoadGemma4MTPDrafter` exists and loads the BF16 E2B local asset: `models/gemma4-e2b-mtp-drafter`.
- Internal tests exercise projection-only, synthetic q-only, real-asset contract, one-step, and multi-step MTP flows.
- The 31B 4-bit assistant asset is present at `models/gemma4-31b-it-mtp-assistant-4bit`, but the drafter loader currently rejects it because it only loads F32/BF16 tensors and the 31B assistant is MLX 4-bit (`U32` packed weights plus scales/biases).
- A regression test records this current blocker: `TestLoadGemma4MTPDrafter31B4BitDocumentsCurrentBlocker`.

## Recommended next steps

1. Add a dedicated Gemma4 loader facade that owns nested config normalization, `language_model.*` prefixing, per-layer KV-head selection, and K=V projection rules.
2. Add MLX 4-bit support to `LoadGemma4MTPDrafter` (or a new `LoadGemma4MTPDrafterMLX`) for packed assistant weights: embeddings, pre/post projection, q_proj/o_proj, MLP projections, and norms.
3. Add a CPU/on-the-fly flag separate from `-gpu`, so huge 4-bit models can load without forcing GPU layer upload.
4. Probe GPU layer residency incrementally after the compact LM-head path: start with `-gpu-layers 4`, then estimate/upload 8, 12, etc. against RTX 3060 VRAM.
5. Wire the 31B MTP assistant path only after normal 31B decode is stable and the 4-bit MTP drafter loader has parity smoke coverage.
