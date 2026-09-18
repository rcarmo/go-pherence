# Checked Whisper word timing and speaker attribution — 13 September 2026

## Result

Go now exposes an opt-in checked Whisper word-timing path and preserves those timings as a separate window/transcript timeline. Community-1 speaker output maps each checked word to the exclusive-turn speaker with greatest positive overlap; exact ties and uncovered words remain unlabelled. Plain segment transcript/VTT output is unchanged. This implements the mechanism but does not substitute for broad annotated SA-WER qualification.

## Numerical contract

- The alignment pass uses a fresh decoder state from the same encoder output.
- It teacher-forces the complete `SOT + language + transcribe + no-timestamps` prompt and generated text tokens.
- Only validated, immutable `alignment_heads` from generation metadata are observed.
- Attention is observed from the CPU path actually used for that decoder pass; no GPU diagnostic values are fabricated.
- The implementation matches Transformers 4.57.1's population normalization over tokens, reflect-padded width-7 median filtering, head mean, float32-cost DTW, NumPy first-minimum tie order, 20 ms frame scale, and final-token timestamp duplication.
- Token grouping preserves UTF-8 split across BPE tokens, uses language-aware Unicode grouping for `zh`, `ja`, `th`, `lo`, `my`, and `yue`, and applies OpenAI Whisper's prepend/append punctuation classes.
- The caller supplies the unpadded 10 ms mel-frame count; padded encoder frames are excluded.

## Evidence

With `CGO_ENABLED=0`, `GOMAXPROCS=2`, and `GO_PHERENCE_DISABLE_NVIDIA=1`:

- Pinned Transformers `4.57.1` / PyTorch `2.14.0+cpu` teacher-forced JFK oracle produced token starts `0.00, 1.08, 1.34, 1.70, 2.28, 3.80, 4.70, 5.70, 5.96, 6.42, 6.72, 6.96, 7.24, 8.14, 8.62, 8.98, 9.22, 9.44, 9.70, 9.88, 10.10, 10.60, 10.98` seconds.
- The same independent oracle plus hash-pinned MINDS inputs covers 19 Portuguese, 6 Portuguese, and 5 French words. The Go forced-token alignment matches every word, token span, and boundary within 20 ms. The oracle is reproducible with `transformers-oracle.py`; `multilingual-oracle.jsonl` retains exact inputs, tokens, words, and times.
- One Portuguese oracle word has equal 4.30 s boundaries. The exact zero-duration span is retained in JSON, receives no speaker without positive overlap, and is projected to a one-millisecond cue only when serialising WebVTT.
- `WHISPER_TINY_MODEL_DIR=... WHISPER_JFK_WAV=... go test ./models/whisper -run '^TestWhisperTinyJFKWordAlignment$' -count=1 -v` passed with 22 words, first `0.00`, final end `10.98`.
- Focused matrix/DTW, UTF-8/punctuation/language grouping, validation, window mapping, reconciliation, attribution and word-VTT tests pass.
- Full `models/whisper`, `runtime/speechjob`, and `cmd/audio/speechjobserve` packages pass; focused vet and Linux/ARM64 test compilation pass.

## Scope and holds

Word timestamps are opt-in through `profile.word_timestamps`; the Community example opts in. A generation document without alignment heads is rejected before a job runs. Word alignment adds a second CPU decoder pass. No service was started or deployed. No private recording was used. Tiny misrecognises substantial text in two Portuguese clips, so this is alignment parity rather than multilingual WER qualification. Broad multilingual timestamp quality and annotated SA-WER/DER/JER remain release blockers.
