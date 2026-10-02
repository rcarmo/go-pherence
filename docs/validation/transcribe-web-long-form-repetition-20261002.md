# transcribe-web long-form repetition fix (2026-10-02)

The Whisper original-Q5 Vulkan deployment (`2dc4a9df`) passed short-file,
Portuguese and two-speaker checks. The first long-file job (26.5 min podcast,
54 windows) exposed two failures caused by the whisper.cpp rolling
previous-text prompt that `OriginalWindowCompatibility` enables.

## Failures

| Run | Result |
|---|---|
| First podcast job | Failed at window 51: `split PCM offset 270000 depth 4: Whisper generation reached its token limit without EOT`. Window 51 hit the token limit, and so did every split leaf down to depth 4. |
| Same audio after the history-drop fix | Completed, but windows 24–29 (720–886 s) repeated "So I was able to do this in the next phase." 34× and "And then we're going to do this in the next phase." 14× |
| 690–900 s clip alone | Coherent text, no repeats, so the loop is seeded by the prompt and not by the audio |

The audio level in the looping region is a steady −17 dB, so it is speech, not silence.

## Change

`decodeDroppingHistoryOnLimit` (`model/whisper/pcm_transcribe.go`) applies
only when a rolling prompt is present, so runs without window compatibility
are unchanged. It decodes the window once more, greedily, without the prompt,
when the first decode:

- hit the generation limit (whisper.cpp marks this a failed decode);
- has token entropy < 2.4 over the last 32 generated tokens, with more than
  32 tokens (`entropy_thold`, the whisper.cpp formula);
- has a zlib compression ratio > 2.4 on the decoded text (OpenAI Whisper
  `compression_ratio_threshold`);
- has three consecutive identical non-empty segments (a short loop that
  stays below the ratio).

whisper.cpp stops conditioning on history at fallback temperatures ≥ 0.5,
and OpenAI Whisper resets the prompt in the same case. The retry is the
deterministic greedy equivalent. Its result is accepted, as both references
accept their best decoder, and the history restarts from that window. Split
leaves use the same helper.

On this podcast, the entropy test alone did not fire: timestamp tokens
between repeats keep entropy near ln 16. Compression ratio fixed the long
loop; the segment-run test fixed a 5× "I'm not an N8N expert." loop at 780 s.

## Verification (production 8093, runtime `cec10a6b…`)

| Check | Result |
|---|---|
| JFK (`asr-en-wav`) | Cues byte-identical to `fast6-jfk.json` (1/1), 3.32 s |
| PT2 (`asr-pt-wav`) | Cues byte-identical to `fast6-pt2.json` (6/6), 6.87 s |
| Podcast 1590 s | Complete, 427 cues, no text repeated more than 2×; 775–790 s matches the standalone clip; 302.6 s wall time (real-time factor 0.19) |
| Cancel at 15/54 windows | Drained to `cancelled`; 15 window acknowledgements, input and decode stage retained |
| `retry-queued` resume | Attempt 2 completed; cues identical to the uninterrupted run (427). Wall time 290 s, because window compatibility replays earlier windows to rebuild the prompt |
| SSE `/progress` (`Accept: text/event-stream`) | `event: progress` with `asr-windows` completed/total |
| Browser (Playwright, Chromium) | Upload `jfk.m4a` with Auto language; row Ready in 4.3 s; "Download transcript" gave `jfk.transcript.vtt` with the expected text; detail view rendered; no console errors |
| Resources | Peak 4.53 GB under MemoryMax 8 GiB; NRestarts 0; existing manifests unchanged on every deploy |
| Gates | `go build ./...`, vet, full `go test ./...`, race (`model/whisper`, `runtime/speechjob`, `cmd/audio/speechjobserve`), arm64/riscv64 builds |

## Known limits

All three limits recorded here were fixed afterwards: seek windows (Go had
been dropping speech at 30 s boundaries), prompt-based resume, and retry
after an engine update. See
[transcribe-web-seek-resume-rebind-20261002.md](transcribe-web-seek-resume-rebind-20261002.md).
