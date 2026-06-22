# MiniCPM-V / MiniCPM-O support

This note tracks support for OpenBMB MiniCPM-V and MiniCPM-O checkpoints from <https://github.com/openbmb/MiniCPM-V>.

## Current scope

Implemented:

- Hugging Face config parsing for the MiniCPM-V/MiniCPM-O family in `loader/config`.
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

- Processor sidecar metadata loading from `preprocessor_config.json` or `processor_config.json`, including nested `image_processor` fields, square resize size, mean/std, rescale factor, patch size, and image sequence length.
- Tokenizer sidecar metadata loading from `tokenizer_config.json` and `tokenizer.json`, including chat-template byte count and MiniCPM image sentinel token IDs when present.
- Special-token resolver that merges config/tokenizer metadata into the prompt planner's image patch/start/end token IDs.
- Safetensors header/index inventory for local checkpoints.
  - Classifies tensors as text embeddings/layers/LM head, vision tower, resampler, projector, norm, or other.
  - Reports metadata readiness without loading tensor payloads.
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

- Full EVA02/SigLIP vision tower tensor loading/execution.
- Resampler tensor loading/execution beyond tensor-name inventory.
- Applying chat templates and tokenizing natural-language conversations end-to-end.
- MiniCPM/Qwen2/Mistral text-backbone weight mapping for MiniCPM-V/O checkpoints.
- End-to-end image+text generation command.
- GPU/SIMD parity gates for the vision tower and resampler.

The committed scaffold deliberately makes the checkpoint and prompt contracts testable before tensor execution is wired in.
