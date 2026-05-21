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

Initial normalization and K=V support are in place, but the next cleanup should extract Gemma4-specific config and weight loading into an explicit loader boundary.

## Recommended next steps

1. Add a dedicated Gemma4 loader facade that owns nested config normalization, `language_model.*` prefixing, per-layer KV-head selection, and K=V projection rules.
2. Add a CPU/on-the-fly flag separate from `-gpu`, so huge 4-bit models can load without forcing GPU layer upload.
3. Probe GPU layer residency incrementally after the loader boundary is stable: start with `-gpu-layers 1`, then estimate/upload 4, 8, 12, etc. against RTX 3060 VRAM.
4. Wire the 31B MTP assistant path only after normal 31B decode is stable.
