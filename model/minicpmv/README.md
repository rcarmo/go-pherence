# MiniCPM-V/O model package

This package owns the Go-side metadata, prompt, preprocessing, and readiness scaffold for OpenBMB MiniCPM-V and MiniCPM-O checkpoints.

## Implemented

- Aggregate metadata loading:
  - `LoadMetadata`
  - `LoadMetadataWithOptions`
- Prompt scaffolds:
  - image placeholder construction
  - MiniCPM-O audio placeholder construction
  - combined image+audio multimodal prompt previews
  - prompt token-span planning for image embedding replacement
- Sidecar-derived contracts:
  - processor metadata to image preprocessing config
  - tokenizer image/audio sentinel token resolution
  - generation defaults passthrough
- Image preprocessing:
  - Go `image.Image` to BCHW `float32`
  - optional square resize
  - rescale/normalize
  - PNG/JPEG decode for CLI inspection
  - patch-grid validation
- Tensor metadata:
  - safetensor name inventory
  - safetensor shape validation
  - explicit safetensors path support
  - resampler binding plan
  - MiniCPM-O audio tensor plan
- Runtime/readiness planning:
  - text execution plan
  - vision/resampler execution plan
  - MiniCPM-O audio execution plan
  - slice-mode plan
  - combined readiness report
  - not-implemented runtime interfaces with `ErrRuntimeNotImplemented`
- Embedding boundary:
  - validated replacement of planned image patch token embeddings with resampler outputs.

## Not implemented yet

- EVA02/SigLIP vision-tower numeric execution.
- Perceiver resampler numeric execution.
- MiniCPM/Qwen2/Mistral text-backbone generation binding for MiniCPM-V/O checkpoints.
- MiniCPM-O audio feature extraction and audio encoder execution.
- End-to-end image/audio/text generation.

## Validation

Use the project-level gate:

```bash
make minicpmv-check
```

For local checkpoint inspection:

```bash
make minicpmv-inspect-model \
  MINICPMV_MODEL=models/minicpm-v-2.6 \
  MINICPMV_SAFETENSORS=models/minicpm-v-2.6/model.safetensors \
  MINICPMV_FLAGS='-require-shapes-ready'
```
