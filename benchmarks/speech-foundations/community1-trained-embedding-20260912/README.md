# Experimental trained Community-1 embeddings

The Go-owned PCM → WeSpeaker Fbank → ResNet34 → masked statistics pooling → embedding path passes all tested raw-embedding endpoint comparisons. Its frontend, trunk and soft-mask support still fail strict reference checks. The new wrapper is explicitly experimental; no service or model default changes.

## Checkpoint and offline conversion

- Model: `pyannote/speaker-diarization-community-1`, revision `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`.
- Raw embedding checkpoint: SHA256 `6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929`, 26,646,242 bytes.
- Inventory:218 tensors, comprising182 F32 learned/statistical tensors and36 I64 BatchNorm counters. Architecture: ResNet34 BasicBlock stages `[3,4,6,3]`,32 base channels,80 mel bins,256-dimensional projection, TSTP pooling and no second embedding layer.
- The raw checkpoint has no `hyper_parameters`. The exporter explicitly selects the checksum-pinned constructor defaults: mono16k,25ms Hamming windows,10ms hop,80 mel bins, snipped edges, FFT power-of-two padding, no dither/energy and whole-window centering. Exact state_dict loading verifies the architecture inventory; it cannot supply missing alternative frontend metadata.
- Converted manifest SHA256: `b6b6081921bbe1b22db78a67b4692c9a5fc0434b0cfa419b818d056e5a8f8ffb`. Two clean exports produce six byte-identical files: weights, four reference traces and manifest.

The exporter validates raw/source/public-PCM hashes, requires a new output directory, uses `weights_only=True` with exactly three pinned pyannote metadata globals, and never downloads weights. It preserves F32/I64 dtypes. Reference execution is one-thread CPU with MKLDNN disabled. Manual trunk traversal matches the source wrapper exactly; pooled projection and shared-trunk masked execution match the complete reference forward exactly.

The same pinned model card declares CC-BY-4.0 for the separately obtained weights. `NOTICE` records attribution and conversion changes. Implementation source licences remain separate, including the existing Apache-2.0 ResNet adaptation. Converted model weights, PCM and raw traces remain outside source control; only metadata/hashes and result logs are included here.

## PCM wrapper and frame ownership

`ExperimentalEmbedding` retains the checked ResNet model. Its80-bin frontend requirement permits reduced-width test models; the constructor does not establish trained checkpoint identity. Callers must verify weight hashes and frontend provenance.

`ForwardPCMFrames` and `ForwardPCMFramesObserved` compute Fbank and all16 residual blocks once for a complete immutable mono16k window of400–160000 samples. No resampling, padding, silence skip or global identity decision occurs. Whole-window centering makes arbitrary chunking non-equivalent.

The returned `EmbeddingPCMFrames` has private feature storage and a producing-wrapper identity. `Shape` exposes value metadata only. `EmbedFrames` and `EmbedFramesObserved` reuse that trunk for multiple mask sets, rejecting frames from another wrapper before pooling. No CNN recomputation occurs on these calls. Pooling uses speaker-major masks and the existing nearest-resize rule, followed by raw unnormalised projection. Empty masks produce the projection bias; callers must apply minimum-speech/overlap admission before using them.

Weights and frame features are immutable; outputs and scratch are call-owned. Observers are synchronous read-only transient views. Cancellation returns no partial result, though completed observer calls remain visible. No driver, process, file or worker pool is opened by the wrapper.

## Trained numerical results

Inputs are digital silence, deterministic broadband noise, one second of public speech (6–7s), and five seconds of the same pinned public sample (6–11s). Each case runs two paths: reference Fbank → Go trunk and Go PCM/Fbank → Go trunk. Each trunk is reused for unweighted pooling and a three-row soft mask including an empty row.

The existing frontend gate is `2e-4` absolute. Trunk boundaries retain the prior synthetic gate `2e-6 + 2e-6*abs(reference)`. The trained pooled-statistics/raw-embedding gate is `2e-4` absolute. Support sums use `1e-6` absolute, and positive-weight frame counts must match exactly. These are bounded fixture checks, not ratified DER or global speaker-quality gates.

| Input | Fbank max error | Go-PCM unweighted embedding | Go-PCM masked embedding | Failed strict comparisons per run |
|---|---:|---:|---:|---:|
| silence-1s | 9.53674e-7 | 3.94881e-7 | 1.10734e-6 | 19/47 |
| broadband-1s | 1.84059e-4 | 9.76957e-7 | 2.36183e-6 | 21/47 |
| public-6–7s | 1.12414e-4 | 9.98378e-7 | 8.60542e-7 | 23/47 |
| public-6–11s | **4.01020e-4** | 9.46224e-7 | 3.09944e-6 | 25/47 |

Three normal runs produce564 comparisons over96,829,152 values:300 pass and264 fail. All48 raw-embedding endpoint comparisons pass, with maximum error `3.0994415283203125e-6`. An additional forced AVX2/FMA-off run also passes all16 endpoints. Shared-trunk support counts are exact; soft-mask sums differ by1.90735e-6 for the one-second cases and1.14441e-5 for five seconds, exceeding their stated gate.

`TestCommunity1TrainedEmbedding` always enforces embedding endpoints and exact support counts while recording every intermediate. `GO_PHERENCE_TEST_COMMUNITY1_EMBEDDING_STRICT=1` additionally enforces all comparison gates and fails all four cases. The strict Make target intentionally returns nonzero. No tolerance has been widened.

A candidate BatchNorm shift/output FMA change produced the same aggregate pass count and endpoint errors. It was reverted; the experiment log is retained. This checkpoint changes no underlying Fbank, CNN, BatchNorm or pooling arithmetic. Segmentation's prior two trained and six synthetic failures are unaffected.

## Verification and review

- `make speech-foundations-check speech-sincnet-fma-check speech-media-integration`: pass, including FFmpeg integration.
- Community-1, audio and `backends/simd/...`:668 passing events /293 top-level passes; six explicit model/strict/diagnostic test skips plus the existing no-tests package skip.
- Thirty shuffled wrapper repetitions:90pass, no skips/failures. Forced AVX2/FMA-off wrapper checks:3pass.
- Trained endpoint target:15pass events across three runs (12cases +3parents); separate fallback:5pass. Strict target retains four failing cases and parent failure.
- Three exporter safety/metadata tests pass without model execution.
- Tests cover scalar/SIMD composition, feature owner mismatch, private immutable feature storage, output ownership, mask reuse, invalid inputs/modes, cancellation in frontend/trunk/pooling and sampled internal checkpoints, recovery, and concurrent pooling of shared frames.
- Affected source vet passes. Community-1 arm64 test binary cross-builds; it was not executed on arm64.
- Whole-tree build retains the exact saved baseline errors. Race compilation still lacks `gcc`; concurrency tests are not race qualification.

A delegated source-only wrapper review found no concrete feature-ownership or cancellation flaw. It requested consistent wrapper-level mode validation and named frontend constants; both changes and regression tests are included. Its request to enforce production channel widths was not adopted: reduced-width80-bin models are intentionally supported, and the constructor documentation now states that it does not authenticate checkpoint identity. Review scope excludes full model numerics/performance/security qualification.

## Reproduce

Use a coordinated CPU window and the previously pinned offline reference environment. Keep the converted cache outside the repository.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/path/to/diar-export/bin/python scripts/community1_embedding_export.py \
  --checkpoint /path/to/3533c8/embedding/pytorch_model.bin \
  --public-wav /path/to/pinned/pyannote-sample.wav \
  --output-dir /path/to/new/embedding-cache

export GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR=/path/to/new/embedding-cache
GOMAXPROCS=2 CGO_ENABLED=0 make speech-community-embedding-check
# Expected failure until frontend/trunk/support gaps are corrected:
GOMAXPROCS=2 CGO_ENABLED=0 make speech-community-embedding-strict
```

[Evidence](evidence.json) and [manifest](manifest.json) retain hashes, stage metrics, strict failures and the rejected arithmetic experiment. Full-file segmentation/embedding mask orchestration, global clustering/DER, service workflow and combined-job performance remain unfinished. No timing claim, GPU work, go-264 merge, push, deployment or restart occurred. LLM/speech services remained inactive under compute window `whisper-community-embedding-0952`.
