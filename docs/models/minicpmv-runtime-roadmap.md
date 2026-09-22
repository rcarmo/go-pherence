# MiniCPM-V/O runtime roadmap

This roadmap starts from the `minicpmv-scaffold-v1` state. Metadata, prompt, preprocessing, tensor inventory, readiness, inspection, and correctness-first CPU text, vision, resampler, and MiniCPM-O audio slices are implemented and gated by `make minicpmv-check`; released-model parity and end-to-end multimodal generation remain pending.

## Current support boundary

Implemented:

- Config parsing for MiniCPM-V and MiniCPM-O variants.
- Processor/tokenizer/generation sidecar loading.
- Image/audio sentinel token resolution.
- Image/audio/multimodal prompt placeholder and token-span planning.
- PNG/JPEG image preprocessing to BCHW `float32`.
- MiniCPM-O audio metadata and feature-frame planning.
- Safetensor inventory, dtype/rank/byte summaries, and shape validation.
- Text, vision, resampler, and audio execution plans.
- Image/audio embedding injection boundary helpers.
- Runtime interfaces that return `ErrRuntimeNotImplemented` for unbound stages.
- Owned-F32 MiniCPM, Qwen2, and Mistral dense text binding with request-local KV state, one-token hidden/logit execution, embedding-prefix greedy decoding, GQA/RoPE/RMSNorm/SwiGLU, Qwen2 Q/K/V bias handling, MiniCPM embedding/depth/logit scaling, and tied/untied LM heads.
- Owned-F32 SigLIP patch embedding and transformer execution with full attention, affine LayerNorm, tanh-GELU MLP, positional embeddings, and synthetic token output.
- Owned-F32 perceiver resampler execution with optional vision-to-language KV projection, packed Q/K/V attention weights, source/query 2D positions, cross-attention, final projection, and composition into image embedding injection.
- Synthetic tests for the three text variants, SigLIP vision, resampler, and injection slices, including malformed shapes/policies/state, ownership, determinism, and non-finite input/weights.
- `minicpmvinspect`, fixture helpers, capability/status reports, and Makefile gates.

Not implemented:

- Pinned independent released-checkpoint text hidden/logit parity.
- Sampling policies beyond deterministic greedy decoding.
- Pinned independent released-checkpoint SigLIP vision parity for both nested and fused-QKV/timm layouts.
- Pinned independent released-checkpoint perceiver resampler/KV-projection parity.
- Pinned independent released-checkpoint MiniCPM-O audio frontend/encoder/projector parity.
- End-to-end generation and parity gates.

## Runtime implementation order

1. **Text backbone binding — CPU/synthetic slice complete**
   - Maps MiniCPM/Qwen2 `llm.*` and legacy Mistral root text weights into an owned-F32 CPU decoder.
   - Validates exact embeddings, attention projections and biases, MLP projections, norm, and LM-head shapes.
   - Runs one-token synthetic hidden-state/logit tests without vision/audio and supports deterministic greedy decode from injected embeddings.
   - Remaining gate: approved, pinned, independently generated released-checkpoint hidden/logit fixtures.

2. **Vision tower execution — CPU/synthetic slices complete**
   - Both nested SigLIP (`vpm.encoder.layers.*`) and the MiniCPM-V 2.0 timm SigLIP fused-QKV layout (`vpm.blocks.*`) implement patch embedding, positional embedding, transformer blocks, and post-LayerNorm output.
   - `PreprocessImageFile` and `BuildVisionExecutionPlan` remain the input/plan boundary.
   - Synthetic image tensor execution is covered. An approved independent real-image feature checksum fixture remains open.

3. **Resampler execution — CPU/synthetic slice complete**
   - Binds resampler query, optional query position, packed attention projection, KV projection, norms, output projection, and final projection tensors.
   - Uses `BuildResamplerTensorPlan` and `NewResamplerShape` as the readiness boundary.
   - Synthetic vision-token to `num_query × hidden` shape, source/query positions, ownership, and deterministic execution are covered. Approved released parity remains open.

4. **Image embedding injection — synthetic composition complete**
   - `VisionEmbeddingCPU` feeds SigLIP and resampler outputs into `InjectImageEmbeddings` over planned spans.
   - Synthetic token embedding replacement is covered without mutating caller-owned embeddings.

5. **MiniCPM-O audio frontend and encoder — CPU/synthetic slice complete**
   - Reuses the exact Transformers-compatible Whisper log-mel frontend for mono 16 kHz PCM.
   - Binds the `apm.*` Whisper encoder and `audio_projection_layer.*` tensors using owned F32 storage, then executes exact GELU, full attention, the two-layer ReLU projector, and stride-2 average pooling.
   - Synthetic feature/PCM-to-audio-embedding shape, policy, ownership, determinism, malformed/non-finite, and pooling tests are covered. Approved released short-audio checksum parity remains open.

6. **Audio embedding injection — synthetic composition complete**
   - `AudioCPU.EncodeAndInject` feeds pooled encoder outputs into `InjectAudioEmbeddings` over spans produced by `BuildAudioPromptPlan`.
   - Synthetic non-aliasing replacement and exact prompt-token count checks are covered; mixed image+audio end-to-end generation remains open.

7. **End-to-end generation**
   - Wire tokenizer/chat template application, image/audio preprocessing, embedding injection, text prefill, and decode.
   - Add strict fixture and parity gates before changing `RuntimeStatusPending` or runtime capabilities.

## Required gates before claiming runtime support

Before any capability flips from false to true:

```bash
make minicpmv-version
make minicpmv-support-summary
make minicpmv-pending-runtime
make minicpmv-check
go test ./model/minicpmv ./cmd/minicpmvinspect -count=1
```

Before claiming end-to-end generation:

- Add at least one committed or documented local fixture for MiniCPM-V image+text.
- Add at least one committed or documented local fixture for MiniCPM-O audio or audio+image prompt metadata.
- Make `minicpmvinspect -require-runtime-ready` pass for the relevant checkpoint class.
- Update `CurrentCapabilities`, `CurrentSupportSummary`, the coverage manifest, and the generated coverage snapshot in the same change.
