# Whisper diarized VTT status

`cmd/diarize-vtt` is the current purpose-built speech command for long-form translated WebVTT output. It is optimized for the local `whisper-large-v3-hf` assets and uses the large-v3 translate prompt by default.

## Default profile

```bash
go run ./cmd/diarize-vtt \
  -input /path/to/audio.m4a \
  -output transcript.vtt
```

Important defaults:

| Flag | Default | Notes |
|------|---------|-------|
| `-model` | `models/whisper-large-v3-hf/model.safetensors` | `tokenizer.json` is loaded from the same directory. |
| `-size` | `large-v3` | `tiny` weights are not bundled/used for this workflow. |
| `-task` | `translate` | Uses Whisper's automatic translation task. |
| `-language` | `pt` | Override with `-language es`, `en`, etc. for source-language prompting. |
| `-chunk` | `10` | Maximum VAD-packed chunk duration in seconds. |
| `-overlap` | `1` | Context padding around VAD-packed cues. |
| `-vad-pack` | `true` | Pack VAD speech regions instead of fixed windows. |
| `-workers` | `min(16, runtime.NumCPU())` | 16 workers was the best local stress setting; 20 regressed. |
| `-max-tokens` | `40` | Tuned decoder cap for short translated cues. |
| `-tokens-per-sec` | `4` | Dynamic per-cue token budget; short cues get smaller caps. |
| `-gpu` | `true` | Uses GPU-assisted encoder, cross-KV precompute, and LM head when available. |
| `-progressive` | `true` | Rewrites the VTT after each completed chunk. |
| `-resume` | `true` | Loads an existing partial VTT and skips covered intervals. |

## What is accelerated today

The production/default path is **GPU-assisted**, not fully GPU-resident:

- GPU encoder linear projections via NVIDIA SGEMM.
- GPU decoder cross-K/V precompute via SGEMM.
- GPU LM-head projection.
- CPU decoder layers, MLPs, self-attention, and cross-attention remain the main bottleneck.
- CPU decoder attention score dots use SIMD; value accumulation uses SIMD SAXPY.

Experimental paths exist but are disabled by default because they are slower on the current workload:

| Env flag | Status |
|----------|--------|
| `GO_PHERENCE_WHISPER_GPU_DECODER_MLP=1` | Resident GPU decoder MLP weights; rejected as default because per-token launches regressed stress RTF from ≈0.79 to ≈0.83. |
| `GO_PHERENCE_WHISPER_GPU_CROSS_ATTN=1` | GPU cross-attention over resident cross-K/V; rejected as default because q/out transfer plus per-layer launches regressed stress RTF from ≈0.68 to ≈0.75. |

## Current performance snapshot

Local 110s stress sample, `large-v3` translate, VAD-packed chunks:

| Configuration | Result |
|---------------|--------|
| 4 workers | RTF≈1.03 |
| 8 workers | RTF≈0.83 |
| 12 workers | RTF≈0.74 |
| 16 workers | RTF≈0.72 before SIMD attention, ≈0.68–0.69 after current decoder optimizations |
| 20 workers | RTF≈0.74 (regression) |

Other relevant findings:

- VAD-packed 10s chunks reduced the stress sample from 13 to 12 chunks and improved RTF from ≈0.83 to ≈0.79 before later decoder optimizations.
- VAD-packed 15s chunks reduced chunk count further but regressed to RTF≈1.26 and was rejected.
- Dynamic token budgets are mostly timing-neutral on the stress sample but reduce hallucinated/repeated tails on short cues.
- Trigram token repetition and repeated/low-value cue filters are enabled for quality.

## Resume/progressive behavior

`diarize-vtt` writes partial output progressively. On resume it:

1. parses the existing VTT;
2. filters stale repeated/low-value cues immediately;
3. sorts cues by timestamp;
4. skips planned jobs by interval coverage, not exact timestamp equality;
5. continues transcribing remaining chunks.

This is important for long files because full `large-v3` translation can still be interrupted before completion.

## Diarization status

Despite the command name, current speaker labels are still a **single-speaker fallback**:

```go
labels := make([]int, len(vad)) // ECAPA weights are not bundled yet; single-speaker fallback.
```

The output uses WebVTT voice tags (`<v Speaker 1>...`) and VAD segments, but real multi-speaker identification is not active until an ECAPA/pyannote-style speaker embedding model is loaded and clustered. Current VTT files should therefore be interpreted as translated speech cues with placeholder speaker labels.

## MTP/speculative status for Whisper

Whisper speculative/MTP scaffolding is correctness-only today:

- `DecoderState` tracks `LastToken` correctly.
- `SpeculativeDecodePrompt` supports explicit language/task prompts.
- `ForwardTokens` and `VerifyDraftSequential` expose the required G+1 verifier shape.
- `AcceptDraftTokens` implements greedy accepted-prefix plus bonus-token semantics.
- The verifier is still sequential internally; no speedup is expected until a fused/batched verifier or smaller acceptable drafter lands.
- `tiny` weights were removed, so there is no bundled small Whisper drafter.

## TurboQuant/KV status

TurboQuant is not useful for the current chunked Whisper workflow. Decoder state is reset per VAD-packed chunk; self-KV is only about 13 MiB/chunk at 40 tokens for large-v3, while cross-KV and decoder compute dominate. TurboQuant may become relevant only if true long-running streaming decode with persistent KV is added.

## Remaining high-impact work

1. Fused/resident decoder kernels that avoid per-token/per-layer GPU launch and host-transfer overhead.
2. A true batched Whisper verifier for speculative/MTP decoding.
3. A real speaker embedding/clustering path for multi-speaker diarization.
4. GPU mel kernel body completion; currently PTX structure exists but the fused FFT/mel body remains pending and is not the dominant bottleneck.
