# MiniCPM-V/O runtime roadmap

This roadmap starts from the `minicpmv-scaffold-v1` state. Metadata, prompt, preprocessing, tensor inventory, readiness, inspection, and a correctness-first dense text CPU slice are implemented and gated by `make minicpmv-check`; released-model text parity and full multimodal tensor execution remain pending.

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
- Synthetic tests for the three text variants, malformed shapes/policies/state, ownership, determinism, and non-finite input/weights.
- `minicpmvinspect`, fixture helpers, capability/status reports, and Makefile gates.

Not implemented:

- Pinned independent released-checkpoint text hidden/logit parity.
- Sampling policies beyond deterministic greedy decoding.
- Numeric EVA02/SigLIP vision tower.
- Numeric perceiver resampler/KV projection.
- Numeric MiniCPM-O audio frontend/encoder.
- End-to-end generation and parity gates.

## Runtime implementation order

1. **Text backbone binding — CPU/synthetic slice complete**
   - Maps MiniCPM/Qwen2 `llm.*` and legacy Mistral root text weights into an owned-F32 CPU decoder.
   - Validates exact embeddings, attention projections and biases, MLP projections, norm, and LM-head shapes.
   - Runs one-token synthetic hidden-state/logit tests without vision/audio and supports deterministic greedy decode from injected embeddings.
   - Remaining gate: approved, pinned, independently generated released-checkpoint hidden/logit fixtures.

2. **Vision tower execution**
   - Implement EVA02/SigLIP patch embedding, transformer blocks, and selected output-token extraction.
   - Use `PreprocessImageFile` and `BuildVisionExecutionPlan` as the input/plan boundary.
   - First gate: synthetic image tensor shape smoke, then local real-image feature checksum fixture.

3. **Resampler execution**
   - Bind resampler query, optional position embedding, attention projection, KV projection, norm, and MLP tensors.
   - Use `BuildResamplerTensorPlan` and `NewResamplerShape` as the readiness boundary.
   - First gate: synthetic vision-token to `num_query × hidden` output shape and deterministic checksum fixture.

4. **Image embedding injection**
   - Feed resampler outputs into `InjectImageEmbeddings` over spans produced by `BuildPromptPlanFromSummary`.
   - First gate: token embedding replacement parity with synthetic placeholders.

5. **MiniCPM-O audio frontend and encoder**
   - Implement or reuse mel/filterbank extraction consistent with the checkpoint `audio_config`.
   - Bind audio encoder tensors using `BuildAudioExecutionPlan` and validate with `ValidateTensorShapes`.
   - First gate: synthetic audio feature tensor to audio embedding shape; later add real short-audio checksum fixture.

6. **Audio embedding injection**
   - Feed audio encoder outputs into `InjectAudioEmbeddings` over spans produced by `BuildAudioPromptPlan`.
   - First gate: mixed image+audio replacement counts via `BuildMultiModalEmbeddingPlan` plus embedding copy checks.

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
