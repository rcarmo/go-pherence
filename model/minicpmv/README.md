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
  - capability summary with `RuntimeStatusPending` until released parity and multimodal tensor execution land
  - text execution plan
  - vision/resampler execution plan
  - MiniCPM-O audio execution plan
  - slice-mode plan
  - combined readiness report
  - staged runtime interfaces with `ErrRuntimeNotImplemented` for unbound stages
- Text CPU reference:
  - owned-F32 MiniCPM/Qwen2 `llm.*` and legacy Mistral root tensor binding
  - exact RMSNorm, causal GQA/RoPE, SwiGLU, Qwen2 Q/K/V biases, MiniCPM embedding/depth/logit scaling, and tied/untied LM heads
  - request-local clonable KV state, one-token hidden/logit output, and deterministic greedy decode from injected embeddings
  - synthetic coverage for all three variants; released-checkpoint parity remains pending
- SigLIP vision and perceiver-resampler CPU reference:
  - owned-F32 `vpm.embeddings`, `vpm.encoder.layers.*`, `vpm.post_layernorm`, and official `resampler.*` tensor binding
  - Conv2D patch embedding, dynamic positional selection, full multi-head self-attention, pre-LayerNorm residual blocks, tanh-GELU MLP, optional vision-to-language KV projection, resampler cross-attention, and final language projection
  - composition through `VisionEmbeddingCPU` into the existing non-aliasing image embedding injection boundary
  - synthetic patch/token/resampler/injection execution with strict shape, policy, ownership, determinism, and non-finite checks; EVA02 and released-model vision parity remain pending
- Embedding boundary:
  - validated replacement of planned image patch token embeddings with resampler outputs
  - validated replacement of planned MiniCPM-O audio patch token embeddings with future audio outputs
  - combined image+audio replacement-count planning before numeric runtime integration.

## Not implemented yet

`CurrentCapabilities().RuntimeStatus` is `RuntimeStatusPending` / `tensor_execution_pending` until these steps land:

- Capture pinned independent released-model hidden/logit parity for MiniCPM/Qwen2/Mistral text backbones.
- Add sampling policies beyond deterministic greedy decoding.
- Execute the legacy EVA/timm vision tower and capture independent released vision/resampler parity.
- Inject MiniCPM-O audio embeddings into the text backbone.
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
  MINICPMV_MODEL=checkpoints/minicpm-v-2.6 \
  MINICPMV_SAFETENSORS=checkpoints/minicpm-v-2.6/model.safetensors \
  MINICPMV_FLAGS='-require-shapes-ready'
```

Use `-strict` only when metadata sidecars and tensors are both present. Tensor-only fixtures should use `-require-tensors-ready -require-shapes-ready`.
