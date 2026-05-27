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

Speaker labels are still a **single-speaker fallback by default**:

```bash
go run ./cmd/diarize-vtt -input meeting.m4a -output meeting.vtt
# speaker model not set; using single-speaker fallback
```

The command now has an opt-in ECAPA path:

```bash
go run ./cmd/diarize-vtt \
  -input meeting.m4a \
  -output meeting.vtt \
  -speaker-model models/speaker-ecapa-voxceleb.safetensors \
  -speaker-threshold 0.7
```

The output uses WebVTT voice tags (`<v Speaker 1>...`) and VAD segments. Real multi-speaker identification requires a converted ECAPA safetensors file; without `-speaker-model`, current VTT files should still be interpreted as translated speech cues with placeholder speaker labels.

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

## Candidate faster Whisper models

The current local model is full `openai/whisper-large-v3` layout (`32` decoder layers). Hugging Face metadata shows compatible safetensors/tokenizer layouts for these faster alternatives:

| Model | Decoder layers | Notes |
|-------|----------------|-------|
| `openai/whisper-large-v3-turbo` | 4 | Same `d_model=1280`, 32-layer encoder, large-v3 tokenizer IDs (`translate=50359`, `transcribe=50360`, `notimestamps=50364`). Downloaded/shape-valid locally; stress RTF≈0.54–0.56. Tokenizer byte decoding and `generation_config` suppression are fixed, but current Go decode still produces source-language Spanish instead of English translation, so it is **not** a default replacement yet. |
| `distil-whisper/distil-large-v3` | 2 | Same large-v3 dimensionality and tokenizer IDs; potentially faster but distillation quality/translation behavior must be validated on the target meeting audio. |
| `distil-whisper/distil-large-v3.5` | 2 | Same shape class; newer distil checkpoint, also needs quality validation. |

All three publish `config.json`, `generation_config.json`, `model.safetensors`, `preprocessor_config.json`, and `tokenizer.json`, so they should fit the existing loader with config selection and tensor-name validation work. `openai/whisper-large-v3-turbo` is present locally under `models/whisper-large-v3-turbo-hf`; tensor names include decoder layers `0..3` as expected. Before enabling it, compare Go logits/task behavior against a Transformers reference for `task=translate`, because the current Go path appears to transcribe Spanish rather than translate to English. The upstream model card documents `generate_kwargs={"task": "translate"}` support, so this remains a Go parity issue until disproven. Use `scripts/whisper_transformers_reference.py` when the optional Python reference stack is available.

## Candidate speaker embedding models

`cmd/diarize-vtt` currently assigns a single speaker label. Candidate models for real multi-speaker labels:

| Model | Format/library | License/notes | Fit for Go pipeline |
|-------|----------------|---------------|---------------------|
| `speechbrain/spkrec-ecapa-voxceleb` | SpeechBrain PyTorch checkpoints (`embedding_model.ckpt`, `hyperparams.yaml`) | Apache-2.0; widely used ECAPA-TDNN speaker verification model. | Best architecture match to the existing `models/speaker/ECAPA` scaffold, but requires checkpoint conversion/parsing from SpeechBrain `.ckpt` to a Go-loadable format. |
| `speechbrain/spkrec-xvect-voxceleb` | SpeechBrain x-vector TDNN checkpoints | Apache-2.0; simpler TDNN/x-vector style. | Potentially simpler than ECAPA; still requires checkpoint conversion. |
| `microsoft/wavlm-base-plus-sv` | Transformers `pytorch_model.bin` | WavLM audio-xvector speaker verification. | More complex transformer encoder; less aligned with current lightweight ECAPA code. |
| `pyannote/embedding` | pyannote-audio PyTorch | MIT but often gated/pyannote-dependent in practice. | Strong diarization ecosystem, but config/runtime mismatch with pure Go path. |
| `nvidia/speakerverification_en_titanet_large` | NeMo | Speaker verification/diarization model, but NeMo archive format. | Requires NeMo export/conversion; less direct. |

Recommended first target: `speechbrain/spkrec-ecapa-voxceleb`, because it is Apache-2.0, popular, and closest to the existing ECAPA implementation. The Go side now has a converted safetensors loading contract via `models/speaker.LoadECAPASafetensors`, plus an optional conversion scaffold in `scripts/convert_speechbrain_ecapa.py` for mapping SpeechBrain checkpoint parameter names into the stable tensor names documented in `models/speaker/load.go`. Download the source checkpoint with `make models-download-speaker`, then convert `models/speechbrain-ecapa-voxceleb/embedding_model.ckpt` into `models/speaker-ecapa-voxceleb.safetensors`.

Current parity finding: the real SpeechBrain checkpoint downloads cleanly, but its architecture is not the simplified ECAPA scaffold currently implemented in Go. The checkpoint uses `blocks.0.conv.conv.weight` with shape `[1024, 80, 5]`, Res2Net/TDNN block internals, `mfa.conv.conv.weight` `[3072, 3072, 1]`, ASP tensors such as `asp.tdnn.conv.conv.weight` `[128, 9216, 1]`, and `fc.conv.weight` `[192, 6144, 1]`. `scripts/convert_speechbrain_ecapa.py --preserve-names` can now convert the checkpoint as-is, and `models/speaker.LoadSpeechBrainECAPASafetensors` validates that real SpeechBrain topology contract. Initial Go inference primitives for the Res2Net/TDNN + ASP path are now present and produce a 192-D embedding smoke from the converted checkpoint, but they still need parity validation against SpeechBrain preprocessing/model output before being used for default speaker labels. Use `scripts/speechbrain_ecapa_reference.py` to dump upstream reference embeddings for an audio file, then compare cosine similarity against the Go path once preprocessing is finalized. A first local JFK smoke using the current Go mel normalization produced cosine≈-0.005 versus upstream SpeechBrain, confirming that preprocessing and/or block semantics are not yet parity-correct. If parity remains poor or too slow after preprocessing fixes, choose a simpler x-vector checkpoint instead.

## Remaining high-impact work

1. Fix/validate `openai/whisper-large-v3-turbo` translate behavior against a Transformers reference; keep full large-v3 as default until turbo produces English translation.
2. Fused/resident decoder kernels that avoid per-token/per-layer GPU launch and host-transfer overhead.
3. A true batched Whisper verifier for speculative/MTP decoding.
4. A real speaker embedding/clustering path for multi-speaker diarization, starting with SpeechBrain ECAPA checkpoint conversion.
5. GPU mel kernel body completion; currently PTX structure exists but the fused FFT/mel body remains pending and is not the dominant bottleneck.
