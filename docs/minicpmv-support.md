# MiniCPM-V support

This note tracks support for OpenBMB MiniCPM-V checkpoints from <https://github.com/openbmb/MiniCPM-V>.

## Current scope

Implemented:

- Hugging Face config parsing for the MiniCPM-V family in `loader/config`.
  - Accepts original `omnilmm` / `OmniLMMForCausalLM` style configs.
  - Accepts newer nested `MiniCPMV*` configs with `text_config`, `vision_config`, and `resampler_config`.
- Normalized readiness summary with text dimensions, vision dimensions, resampler shape, image token IDs, slice-mode flag, and runtime note.
- Prompt/image-token planning in `model/minicpmv`.
  - Validates the upstream `<im_start> <im_patch>... <im_end>` contract.
  - Supports patch-only spans for checkpoints that do not use start/end tokens.
  - Reports exact replacement spans for injecting vision/resampler embeddings into the language embedding stream.
- Pure-Go image preprocessing in `model/minicpmv`.
  - Converts arbitrary Go `image.Image` inputs to RGB/NRGBA.
  - Optional square bilinear resize.
  - Rescale + normalize into BCHW `float32` pixel values.
  - Validates patch-grid divisibility and reports the vision patch count before tensor execution.
- Resampler shape validation matching the upstream `Resampler` contract:
  - `num_query = grid_size^2`
  - `embed_dim = language hidden size`
  - default `num_heads = embed_dim / 128`
  - optional `kv_proj` when vision hidden size differs from language hidden size.
- Opt-in model download group:

```bash
make models-download-minicpmv
```

- Local config inspection command:

```bash
make minicpmv-inspect
bin/minicpmvinspect -model models/minicpm-v-2.6
bin/minicpmvinspect -model models/minicpm-v-2.6 -json -require-config-ready
```

Some OpenBMB Hugging Face repositories are gated; set `HF_TOKEN` or `HUGGINGFACE_TOKEN` before downloading when required.

## Upstream execution graph reference

The original OpenBMB code path is:

1. Tokenize prompt containing image sentinels.
2. Preprocess image for the configured vision tower.
3. Run the vision tower, e.g. EVA/SigLIP depending on checkpoint generation.
4. Drop prefix/class tokens when required.
5. Run the one-layer perceiver `Resampler` to produce `num_query` language-sized soft tokens.
6. Replace `<im_patch>` token embeddings between `<im_start>` and `<im_end>` with those soft tokens.
7. Decode with the text LLM backbone.

## Remaining work

Not implemented yet:

- Processor JSON auto-loading for checkpoint-specific image mean/std/resize fields.
- Full EVA02/SigLIP vision tower tensor loading/execution.
- Resampler tensor loading/execution.
- MiniCPM/Qwen2/Mistral text-backbone weight mapping for MiniCPM-V checkpoints.
- End-to-end image+text generation command.
- GPU/SIMD parity gates for the vision tower and resampler.

The committed scaffold deliberately makes the checkpoint and prompt contracts testable before tensor execution is wired in.
