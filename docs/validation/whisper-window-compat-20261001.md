# Original multi-window compatibility — 1 October 2026

`PCMTranscribeOptions.OriginalWindowCompatibility` is a new explicit, experimental option. It applies two whisper.cpp `whisper_full` behaviours across the 30 s windows of one `TranscribePCMWindows` call. With the private `vulkan-original-q5-padded-integer-dot-mmq-tanh` encoder and `OriginalDecoderCompatibility`, PT2 now matches the original on all six segments, including the second-window segment 30.00–35.12 (previously 30.00–43.76). Default output is unchanged.

The attribution is in [the encoder-GELU report](whisper-encoder-gelu-20261001.md#pt2-second-window-attribution). On PT2's second window, the Go decoder reproduced the original's tokens `<|0.00|> E aí <|5.12|>` only with both behaviours applied.

## Behaviour

1. **Whole-clip mel floor.** `log_mel_spectrogram` clamps at the maximum over the whole clip minus 8. A read-only pre-pass computes the raw log10 mel maximum of every planned window with the same arithmetic, then each window uses `floor = max(window max, clip max) − 8`. Values above the floor are unchanged. `loader/audio` exposes `WhisperLogMelMaxContext` and `WhisperLogMelClipFloorContext`; `WhisperLogMelContext` output is unchanged.
2. **Previous-text prompt.** Each later window is prompted with `<|startofprev|>`, the last ≤223 previously decoded tokens (timestamps included) and then SOT/language/task. The history becomes the tokens used in that prompt plus the window's generated tokens, as in `prompt_past1`. It clears when `seek > 0 && seek + 500 ≥ n_len_org` (at most 5 s of audio left). `no_context` in whisper.cpp clears history only at the start of `whisper_full`.

Generation-limit halves share the rolling history in order. VAD runs that call `TranscribePCMWindows` on compacted audio inherit both behaviours within that call; gap-preserving VAD windows are separate calls. Resume (`firstWindow > 0`) is rejected because the history is not persisted. The option requires a tokenizer with `<|startofprev|>` at `transcribe + 2`; otherwise it fails closed.

Known residual difference: the original computes a single STFT over the whole clip, while Go reflect-pads at internal window edges. That affects the frames nearest each window boundary (frame 0 uses 200 reflected samples; frame 1 and frame 2999 use 40), and the clip maximum only if it falls on such a frame.

`Decoder.AdvanceToken` runs a decoder step without the final LayerNorm and vocabulary projection. `decodePCMWindow` uses it for every prompt token except the last, since those logits are never read. This is exact: a synthetic test shows identical state and later logits bits, with and without decoder compatibility. Native outputs on all 10 fixtures × 5 repeats are byte-identical with and without it, in both default and window-compatible modes.

## Results (MMQ-tanh + decoder compatibility, 5 repeats, outputs repeat exactly)

| Fixture | Original | Without window opt-in | **With window opt-in** |
|---|---|---|---|
| JFK | 0–10.40 | 0–10.40 | **0–10.40** |
| PT row 0 | 0–7.36 | 0–7.36 | **0–7.36** |
| PT2 (2 windows) | …, 21.44–22.14, 30.00–35.12 | …, 30.00–43.76 | **…, 21.44–22.14, 30.00–35.12** (all 6 exact) |
| FR row 0 | — | 0–2.42 | 0–2.42 |
| Two JFK groups VAD + words | 2 | 2, text equal | 2, text equal |
| Podcast VAD + words | 7 | 7, text equal | 7, text equal |
| PT1 (whisper-cli reference) | 0–4.58 | 0–4.69 | 0–4.69 |

The single-window fixtures are unaffected apart from cost. Remaining differences are VAD segmentation (Go preserves gaps; the original compacts audio) and references from other harnesses (whisper-cli PT1 and podcast).

Median request seconds:

| Fixture | Without window opt-in | With opt-in, prompt logits computed | **With opt-in + AdvanceToken** |
|---|---:|---:|---:|
| JFK | 4.385 | 4.486 | 4.484 |
| PT row 0 | 4.373 | 4.502 | 4.447 |
| PT2 | 9.210 | 11.030 | **10.465** |
| Podcast VAD + words | 8.256 | 8.359 | 8.321 |

The mel pre-pass costs about 0.04–0.1 s per single-window clip. On PT2 the roughly 70-token previous-text prompt is still decoded token by token on the CPU, about 1 s after skipping its logits. whisper.cpp evaluates the prompt in one batch; batching it belongs with the GPU decoder work.

## Verification and isolation

Unit tests cover:

- clip-floor equality and lifting, silence, maximum consistency and invalid maxima;
- prompt order, the 223-token truncation, rejection of invalid previous tokens and of a missing or misplaced marker;
- the advance path;
- `AdvanceToken` exactness.

`make model-layout-check host-build host-vet host-test docs-check` (541 Markdown files, zero broken links), race for `model/whisper` and `loader/audio`, and ARM64/RISC-V builds pass. Native runs (one build plus 30 fixture runs) used a user-authorised @llama isolation window (CPU4/8GiB/no-swap, physical Intel GPU, Qwen idle, ≤120 s, sequential); all containers exited 0 and were removed.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-window-compat-20261001/) holds benchmark JSON/environments, logs, drive scripts, the original comparison and gate logs.
