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

The 31B main model is about 18 GiB on disk in 4-bit MLX form. CPU dequant-at-load is not viable on this host; a plain CPU load was OOM-killed. Use on-the-fly quantized loading. It is currently enabled by the `llmgen -gpu` path and by the experimental `llmgen -mtp-smoke` path.

## Current smoke status

The packed-MTP runtime smoke loads the main model on the on-the-fly path, loads the 31B assistant as packed MLX 4-bit weights, builds a minimal external-KV view, and runs one q-only drafter step:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/llmgen \
  -model models/gemma4-31b-it-4bit \
  -mtp-drafter models/gemma4-31b-it-mtp-assistant-4bit \
  -mtp-smoke \
  -prompt "Hello"
```

Latest local result on 2026-05-21:

```json
{
  "model_hidden": 5376,
  "model_layers": 60,
  "drafter_hidden": 1024,
  "drafter_backbone": 5376,
  "drafter_layers": 4,
  "packed_embedding": true,
  "packed_projection": true,
  "packed_layer_weights": true,
  "previous_token": 9259,
  "token": 236814,
  "logits_len": 262144,
  "next_activation_len": 5376,
  "load_seconds": 0.262506734,
  "step_seconds": 0.468109107
}
```

The main model load for that run was `16.25s`.

A real-prompt smoke is also available. It prefills the prompt through the main model, captures final activation and float KV using the Generate-equivalent Gemma4 per-layer-input path, maps drafter layers onto compatible main-model KV source widths, and feeds that state into the packed MTP assistant:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/llmgen \
  -model models/gemma4-31b-it-4bit \
  -mtp-drafter models/gemma4-31b-it-mtp-assistant-4bit \
  -mtp-smoke \
  -mtp-real-prompt \
  -prompt "Hi"
```

Latest local result on 2026-05-21:

```text
MTP prompt prefill: 299.06s (10 tokens, next=100)
step_seconds: 0.393279808
wall_seconds: 317.187
```

This proves the real activation/KV handoff, but also shows that CPU/on-the-fly 31B prompt prefill is the immediate bottleneck. A 72-token complex-prompt benchmark on this path would take tens of minutes; long-prompt MTP benchmarking should wait for GPU/prefill acceleration or a much smaller model.

The compact GPU LM-head smoke also loads successfully:

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

`cmd/llmgen` now exposes an experimental `-mtp-smoke` mode for Gemma4 MTP. This is a runtime smoke path, not full speculative generation: it loads `Gemma4MTPDrafter`, runs one q-only drafter step with a minimal external-KV view, and exits. The `-speculative` flag remains the stock-weight proposer/replay verifier path and is separate from Gemma4 MTP.

Current Gemma4 MTP status:

- `LoadGemma4MTPDrafter` exists and loads both the BF16 E2B local asset (`models/gemma4-e2b-mtp-drafter`) and the 31B MLX 4-bit assistant.
- Internal tests exercise projection-only, synthetic q-only, real-asset contract, one-step, and multi-step MTP flows.
- The 31B 4-bit assistant asset is present at `models/gemma4-31b-it-mtp-assistant-4bit` and loads while keeping large matrices packed as MLX 4-bit weights.
- `PreProjectInto`, q/o projections, MLP projections, `PostProjectInto`, and token embedding lookup dispatch through packed MLX helpers when packed weights are present. The only dequantization in the 31B assistant smoke is row-local embedding dequantization and normal small BF16 norm/scalar tensors.
- `llmgen -mtp-smoke` validates the runtime-facing seam and prints shape/timing JSON. With `-mtp-real-prompt`, it first builds real prompt activation/KV via `BuildMTPPromptContext` instead of using zero external KV.
- Regression coverage: `TestLoadGemma4MTPDrafter31B4BitKeepsPackedWeights`.

## Recommended next steps

1. Add a dedicated Gemma4 loader facade that owns nested config normalization, `language_model.*` prefixing, per-layer KV-head selection, and K=V projection rules.
2. Promote the on-the-fly loader switch from `-gpu`/`-mtp-smoke` side effects to an explicit CLI flag for huge 4-bit CPU/runtime experiments.
3. Probe GPU layer residency incrementally after the compact LM-head path: start with `-gpu-layers 4`, then estimate/upload 8, 12, etc. against RTX 3060 VRAM.
4. Convert the smoke seam into full Gemma4 MTP speculative generation: capture real verifier activations/KV, run adaptive multi-draft assistant steps, batch verifier candidates, and commit only accepted KV prefixes plus the bonus token.
