# Long-form Whisper: prompt prefill and word alignment — 5 October 2026

The long-audio gap reported in
[transcribe-web-seek-resume-rebind-20261002.md](transcribe-web-seek-resume-rebind-20261002.md)
(podcast: app 412 s, whisper.cpp greedy 266.6 s) was not a matched comparison:
the app also aligns words, and whisper.cpp does not. A matched stage profile
showed that the per-token decoder was not the cause. The time went to prompt
prefill and to the word-alignment pass. Both are now faster, and outputs are
byte-identical.

## Method

- **Input:** 180 s of dense speech from `testdata/podcast.wav` (600–780 s),
  16 kHz mono PCM, SHA-256 `261dab48…940d34d`. It decodes as 8 seek windows.
- **whisper.cpp:** `whisper-cli` from the pinned Vulkan build with
  `-t 4 -bs 1 -l en`, flash attention on (default) and no word timing.
- **Go:** the `TestWhisperPerformanceGoalArm` harness with backend
  `vulkan-original-q5-padded-integer-dot-mmq-tanh`, original decoder and
  window compatibility, GPU cross K/V and the packed-Q5 CPU decoder. Each arm
  makes 5 requests in one process. Long fixtures need the new explicit
  `GO_PHERENCE_WHISPER_BENCH_LONGFORM=1` opt-in, which raises the limit to 600 s.
- **Guards:** CPU4/8 GiB/no-swap podman containers on the Intel GPU only. The
  run aborts if Qwen on 11434 (`/slots`) or on 8094 (`/health.processing`)
  is busy, or if a transcribe-web job is queued or running.

## Attribution (before)

| Stage, 180 s excerpt | whisper.cpp | Go |
|---|---:|---:|
| Encoder (8 windows) | 22.7 s | 20.15 s |
| Generated tokens | 692 steps, 5.9 s (8.5 ms each) | 701 steps, ~5.45 s (7.8 ms each) |
| Prompt prefill | 1458 tokens, 0.59 s (GPU) | 1454 tokens, **5.4 s** |
| Request, no word timing | 30.0–30.5 s | 31.5 s |
| Request, word timing | — | 37.9 s |

The batched prefill read each Q5 weight row once, but it still dequantised the
row again for every prompt row. That ran at about 21 GMAC/s, roughly a tenth
of the four-core AVX2 peak. The word-alignment pass replayed 658 tokens one at
a time through `ForwardToken`. Each step computed vocabulary logits that were
then discarded, and the observer forced cross attention to run serially.

## Changes

- `simd.SdotQ5_0x4`: an AVX2 kernel that dequantises each Q5_0 block once and
  multiplies it by four input rows. Each row has its own even/odd accumulators
  and runs `SdotQ5_0`'s exact instruction sequence and reduction, so every
  result is bit-identical to `SdotQ5_0`. On one core, four 1280-element rows
  take 296 ns against 630 ns for four single-row calls. The non-AVX2 fallback
  calls `SdotQ5_0`.
- `linearQ5Batch` processes rows four at a time over blocks of 64 output rows.
  Tiling only changes the order in which independent outputs are computed.
- `AdvanceTokens` now accepts a cross-attention observer. The batched path
  calls it with exactly the rows the sequential path produces, possibly
  concurrently for distinct heads. This contract is documented on
  `CrossAttentionObserver`.
- `AlignWordsChecked` feeds the 4-token prefix and the text tokens through one
  `AdvanceTokens` call. It no longer computes logits. The observer takes its
  slot from `tokenPosition` instead of shared loop state.
- Harness: `DecoderStepTokens`, `DecoderPrefillTokens` and
  `DecoderPrefillSeconds` are reported per request.

## Results (after)

Clean rerun with whisper.cpp interleaved between the Go arms:

| 180 s excerpt | Before | After | whisper.cpp |
|---|---:|---:|---:|
| Go, no word timing (median of 5) | 31.5 s | **29.4 s (−6.7%)** | 30.18 s |
| Go, word timing (median of 5) | 37.9 s | **31.1 s (−17.8%)** | no DTW |
| Prompt prefill | 5.4 s | 3.2 s | 0.59 s |

All 5 requests in each arm match the pre-change baseline byte for byte:
segments, text, tokens and word timings. Go without word timing is now 2.5%
faster than whisper.cpp. With word timing it is 3% slower than whisper.cpp
without it.

Standard fixtures, baseline `107c814d` against the new build. Each arm has 5
requests, and all 10 requests per row are byte-identical:

| Fixture | Base | New | Δ |
|---|---:|---:|---:|
| JFK | 2.786 | 2.813 | +1.0% |
| JFK, words | 3.085 | 2.869 | −7.0% |
| PT row 0 | 2.779 | 2.801 | +0.8% |
| PT row 0, words | 3.073 | 2.864 | −6.8% |
| PT2 (2 windows) | 5.965 | 5.883 | −1.4% |
| PT2, words | 6.697 | 6.104 | −8.9% |
| JFK + VAD | 2.843 | 2.831 | −0.4% |
| JFK + VAD, gap-preserving words | 3.138 | 2.911 | −7.2% |
| Groups + VAD | 3.184 | 3.174 | −0.3% |
| Groups + VAD, gap-preserving words | 6.335 | 5.944 | −6.2% |

Single-window runs without word timing have no prompt, so they do not change.
Compact VAD with word timing still fails closed on a word that crosses a
removed gap, with the same error in both builds, as documented before.

## Gates

- Model-free tests:
  - `TestSdotQ5_0x4MatchesSingleRow` covers random rows, special scales,
    subnormal/zero/infinite inputs and padded strides.
  - `TestLinearQ5BatchMatchesPerElement` covers 1–33 rows with 1, 3 and 4
    workers.
  - `TestAdvanceTokensObserverMatchesSequential` compares every
    (layer, head, position) row, with and without compatibility.
  - The existing prefill and alignment tests still pass.
- `gofmt`, `go build ./...`, `go vet ./...` and full `go test ./...` pass.
- Race tests pass in `localhost/go264-ffmpeg:7.1.3` for
  `backends/simd/runtime`, `model/whisper`, `runtime/speechjob/...`,
  `cmd/audio/speechjobserve` and `cmd/audio/transcribe-web`.
- linux/arm64 and linux/riscv64 cross-builds pass, as does the arm64 test-binary
  build.

## Remaining

- Prefill is still 3.2 s on CPU against 0.59 s on whisper.cpp's GPU. The next
  exact step is the prefill cross attention, about 22 GMAC over 1454 rows.
- The full podcast time in the app has to be re-measured after deployment.
