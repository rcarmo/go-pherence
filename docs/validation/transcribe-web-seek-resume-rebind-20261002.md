# transcribe-web: seek windows, prompt resume, engine-update retry (2026-10-02)

This change fixes the three limits left open in
[transcribe-web-long-form-repetition-20261002.md](transcribe-web-long-form-repetition-20261002.md).

## 1. Dropped speech at window boundaries (accuracy)

On the 26.5-minute podcast, Go produced 4,189 words against whisper.cpp's
5,049. Every gap in the Go transcript ended on a 30 s boundary (one ran from
582.2 s to 600 s).

`OriginalWindowCompatibility` used fixed 30 s windows. When speech runs past
a window, Whisper ends with `<|t_end|><|t_next|>` and EOT. whisper.cpp then
seeks to `t_next` (`seek += seek_delta`) and decodes the rest in the next
window. Go discarded everything from `t_next` to 30 s.

The compatibility path now follows whisper.cpp's seek loop
(`transcribePCMSeekFrom`):

- **Next window start.** `seek_delta` is twice the last timestamp above
  `<|0.00|>`. It is a full chunk when there is no timestamp, or when the
  output ends with a single timestamp after text (the last segment closed).
  The single-timestamp case is capped at `seek_end - seek`.
- **Loop exit.** The loop stops at `seek + 10 >= seek_end`, where
  `seek_end = 1 + (n - 200) / 160`.
- **Ownership.** Each window owns `[Start, EmitEnd)`. `EmitEnd` is the next
  window's `Start`.
- **Word alignment.** When a window ends in an unfinished segment, DTW is
  bounded to the seek point, matching whisper.cpp's
  `n_frames = min(3000, seek_delta, …)`. This keeps words inside the
  window's span.
- **Segment and word times.** Times are mapped through integer samples with a
  single division. Without this, `704.94 + 17.82` gave `722.7600000000001`
  while the next window started at `722.76`, which the transcript stage
  rejected as an overlap.
- **Runtime.** The window stage and transcript reconciliation validate a chain
  (`ValidSeekWindow`) instead of fixed-plan geometry. ASR progress is now
  reported in samples, which the UI shows as seconds. The stage identity
  schema is `speechjob-go-whisper-seek-windows-v3`.

Fixed-plan behaviour without compatibility mode is unchanged.

| Podcast, 1590 s | Words | Disagreement with whisper.cpp greedy (`-bs 1 -bo 1 -nf`) | Segments |
|---|---|---|---|
| whisper.cpp greedy | 5,043 | — | 350 |
| Go, fixed windows (before) | 4,189 | 18.9% against whisper.cpp default beam | 427 |
| Go, seek windows | 5,038 | 0.46% (0.0–0.7% per 5 min) | 348 |

The remaining differences are hyphenation and capitalisation ("one off" /
"one-off"), and the segment splits that follow such token flips. Neither
output repeats a segment. Greedy whisper.cpp does not loop at 12:00 either;
the earlier Go loop came from fixed windows feeding the wrong prompts. Disagreement
with whisper.cpp's default beam search is 1.49%.

JFK and PT2 cues and words are byte-identical to the previous runs and to the
qualified harness outputs.

## 2. Resume replayed finished windows

Each window record now persists `prompt`, the last ≤223 rolling-context
tokens. Together with the window's `EmitEnd`, this lets a resume continue at
the next seek with `TranscribePCMWindowsFromPrompt` instead of re-decoding
earlier windows. A record without a prompt still falls back to replay.

| Resume test | Result |
|---|---|
| Fixed windows: cancel at 30/54 | Resume 131.9 s (was 290 s at 15/54); output identical |
| Seek windows: cancel at 823 s (52%) | Drained in 0.3 s; resume 208.5 s; 348 cues and all words identical to the uninterrupted run |

## 3. Recordings from before an engine update

A redeploy changes the profile configuration, so older unfinished recordings
reported `profile_available: false`, and retrying them gave `profile_changed`.

An explicit `retry-queued` now calls `Store.Rebind`. Rebind applies only to a
failed or cancelled recording whose media is retained and which is not
running. It moves the recording to the current configuration of the same
profile ID, drops its checkpoints and removes stale stage and window files.
Plain `enqueue`, completed recordings and unknown profile IDs are refused.

The UI offers "Retry with updated engine". "Upload again" appears only when
the media was released.

Live check: the old failed podcast job was retried from the browser and
completed on attempt 2, with output identical to a fresh run.

## Performance (known gap)

| Podcast, 1590 s | Wall time |
|---|---|
| whisper.cpp greedy, Vulkan | 266.6 s (~3.3 s per window) |
| Go app (media decode, seek windows, word alignment) | 412 s (~5.2 s per window, ~80 windows) |

Go's encoder is ahead of whisper.cpp on the short benchmarks. On dense long
audio the CPU decoder (~7 ms/token) and the DTW word-alignment pass dominate;
whisper.cpp runs its decoder on the GPU and does no DTW by default. This
remains the top item in
[whisper-optimization-map-20261002.md](whisper-optimization-map-20261002.md).

## Gates

- `go build ./...` and vet pass.
- Full `go test ./...` passes.
- Race tests pass for `model/whisper`, `runtime/speechjob/...`,
  `cmd/audio/speechjobserve` and `cmd/audio/transcribe-web`.
- arm64 and riscv64 builds pass.
- Every deploy left existing manifests unchanged, with NRestarts 0 and peak
  memory at or below 4.49 GB.
