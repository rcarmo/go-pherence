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
  - prompt token-span planning for image and MiniCPM-O audio embedding replacement
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
  - capability summary with `RuntimeStatusPending` until tensor execution lands
  - text execution plan
  - vision/resampler execution plan
  - MiniCPM-O audio execution plan
  - slice-mode plan
  - combined readiness report
  - not-implemented runtime interfaces with `ErrRuntimeNotImplemented`
- Embedding boundary:
  - validated replacement of planned image patch token embeddings with resampler outputs
  - validated replacement of planned MiniCPM-O audio patch token embeddings with future audio outputs
  - combined image+audio replacement-count planning before numeric runtime integration.

## Not implemented yet

`CurrentCapabilities().RuntimeStatus` is `RuntimeStatusPending` / `tensor_execution_pending` until these steps land:

- Bind MiniCPM/Qwen2/Mistral text-backbone weights and prefill/decode.
- Execute EVA02/SigLIP vision tower.
- Execute perceiver resampler and KV projection.
- Inject image/audio embeddings into the text backbone.
- Execute MiniCPM-O audio feature extraction and audio encoder.
- Add end-to-end MiniCPM-V/O generation parity gates.

## Validation

A compact committed MiniCPM-O fixture lives under `testdata/minicpmo_fixture` and exercises aggregate metadata, audio metadata, special-token resolution, and multimodal prompt preview construction. Use `LoadMiniCPMOFixtureMetadata` and `LoadMiniCPMOFixtureExpectedSummary` from tests/tools. Its `expected_summary.json` captures stable fixture expectations for future runtime work.

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

Use `-strict` only when metadata sidecars and tensors are both present. Tensor-only fixtures should use `-require-tensors-ready -require-shapes-ready`.
