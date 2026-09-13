# Register-tiled F32 linear candidate

The explicit `NewVkLinearRegTileF32` candidate reduces the tested Turbo encoder median from 14.24 to 8.30 seconds in an isolated confirmation, with bit-exact output. The original `NewVkLinearF32` remains available. After later review of these results, `NewVulkanEncoder` selected the register tile on 13 September 2026; [PROMOTION.md](PROMOTION.md) records that narrow change. Turbo's overall performance target remains unmet.

## Kernel and API

The shader computes a 32×32 output tile with a 32-element reduction tile. Its 16×16 invocations each retain four accumulators and share two activation/two weight values per reduction step. Two 1024-float shared arrays consume 8192 bytes. The existing 16×16 output tile uses 2048 bytes and one accumulator per invocation. Neither kernel changes the X[M,K], W[N,K], bias[N], output[M,N] layout or requires packed weights.

Positive M/N/K remain bounded to 16384. Exact shapes/byte extents, queried device limits and output non-overlap are checked by the shared wrapper. Candidate dispatch uses ceil(N/32)×ceil(M/32); baseline uses ceil(N/16)×ceil(M/16). Output tails are individually guarded; zero-filled row/column/K tails still reach both barriers. The candidate introduces no new SPIR-V opcodes.

The original Whisper test selects the candidate through private `newVulkanEncoderVariant`; the later promotion changes the explicit public Vulkan constructor's internal F32 kernel only. It adds no environment selector or automatic Vulkan device selection. Generic operator stages remain caller-owned as before. This is F32 execution, not F16/quantisation.

## Static, mock and native checks

- All 17 embedded and 17 rebuilt shaders pass static validation; seven rebuild byte-identically, including the candidate. All 17 narrowly normalised comparisons pass. Checker: six tests, 57 assertions.
- Offline selection: 128 top-level tests/438 passing events, zero failures/skips. Thirty shuffled Linear/Plan/Shader repeats: 4710 passing events.
- Three new top-level tests model twelve shapes across four schedules, shared tile/output ownership, exact serial order versus the original source model, dimensions around 16/32 boundaries, K up to 16384, candidate geometry/admission and successful construction/close.
- Native kernel qualification, three repetitions: 36 small numerical cases/15,645 values and three full Turbo shapes. Small fixtures meet the predeclared `2e-5 + 2e-5*abs(reference)` float64 budget, max absolute error `9.660682177781155e-6`.
- Full-size shapes compare candidate and original outputs bit-for-bit. Every timed output is downloaded/checked outside the timer, with arena guard checks. Allocation counts/bytes return to baseline.
- The existing reduced native encoder suite passes 30 events across three repeats, including prior rollback/cancellation/PCM bridge checks.

Read-only shader review found no scoped indexing, weight-orientation, barrier or output-ownership bug. Harness review led to explicitly selecting `minds-pt-0` for encoder timing and hash-pinning the committed transcript baseline. Public speech comparison checks tokens/timestamps, not equality of timing with an older run. One initial compile error in a source-model helper call was fixed before native checks.

## Kernel timing screen

Each full shape uses four alternating ABBA/BAAB blocks, eight warm samples per kernel; three processes yield 144 timed dispatches. Inputs, weights and bias are identical and resident. The median includes host dispatch/submission/fence wait but excludes upload, download and validation. No GPU timestamp attribution is claimed.

| Shape M,K,N | Three-process candidate speedup |
|---|---:|
| 1500,1280,1280 | 2.055–2.064× |
| 1500,1280,5120 | 2.133–2.151× |
| 1500,5120,1280 | 2.125–2.143× |

An exploratory first run was 2.06–2.27×; it is retained separately. These three shapes are the measured Turbo projection families, not hardware-wide or arbitrary-shape performance qualification.

## Complete encoder and speech

Each encoder test creates both baseline and candidate resident encoders from the pinned trained Turbo checkpoint. An initial forward warms/verifies each, followed by two alternating ABBA/BAAB blocks (four samples per encoder). Every final hidden state is bit-exact. Four candidate public transcriptions are compared to the hash-pinned baseline windows/tokens/segment timestamps, which had 0 WER on those fixtures.

| Process | Baseline median | Candidate median | Ratio | Resource qualification |
|---|---:|---:|---:|---|
| Final 1 | 14.235 s | 8.306 s | 1.714× | Setup overlaps another session's CPU diarization; not isolated |
| Final 2 | 14.232 s | 8.413 s | 1.692× | No reported neural overlap; shared monitor includes swap-out |
| Isolated confirmation | 14.244 s | 8.298 s | 1.716× | Explicit exclusive compute window; unchanged swap counters |

An earlier exploratory encoder run was 1.721×. The isolated confirmation is the primary timing result. All final processes preserve exact encoder output and all twelve final public-speech window/token/timestamp comparisons pass. Candidate public requests are roughly 9.2–9.9 seconds; these durations are diagnostics, not balanced public-request A/B claims.

The isolated encoder speedup is material but insufficient for the planned ≥8× ASR target. Attention still needs work. The later promotion makes this the F32 kernel inside the explicit Vulkan encoder; it does not enable Vulkan by default or roll out a service.

## Resource and coordination caveats

Two resident encoders require about 5.28 GB of explicitly tracked native memory/68 arenas, plus host model and opaque driver allocations. The test sets an 8-GiB/80-allocation package cap; this is not a whole-process memory cap.

The initial two-final-process monitoring interval (08:47:00–08:52:22 UTC) sampled MemAvailable down to 18,892,372 KiB and recorded 55 pages of swap-out (`635793→635848`), with swap-in fixed at 36305. It must not be reported as a zero-swap run.

A saved go-264 CPU diarization log spans approximately 08:45:32–08:47:07 UTC. It overlaps the first six seconds of final process 1 (08:47:01 UTC start), during construction/setup and before measured encoder samples. Nevertheless, that process is labelled non-isolated. The neighbouring session acknowledged the admission error and held work for the confirmation.

The exclusive confirmation ran 08:55:28–08:58:09 UTC. Its 155 one-second samples recorded minimum MemAvailable 18,835,472 KiB (about 17.96 GiB); swap-in 36305 and swap-out 635848 stayed unchanged. CPU process summaries are retained in the monitor. No LLM or speech service was restarted. Sampling cannot exclude sub-second pressure or establish power/thermal attribution.

## Reproduce

Use a coordinated compute window and the verified local model/media caches.

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
GO_PHERENCE_TEST_VULKAN_LINEAR_TIMING=1 GOMAXPROCS=2 CGO_ENABLED=0 \
  make speech-vulkan-linear-regtile-check

export GO_PHERENCE_WHISPER_TURBO_DIR=/path/to/verified/whisper-turbo-41f01f3
export GO_PHERENCE_MINDS_FIXTURE_DIR=/path/to/MINDS-cache
export GO_PHERENCE_WHISPER_JFK_PATH=/path/to/pinned/samples/jfk.wav
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-turbo-regtile-check
```

Equivalent native commands were executed; make targets are explicit wrappers. Full speech/affine/media/Vulkan regressions, affected vet and amd64 builds pass. Vulkan arm64 test cross-build passes; Whisper arm64 FFT and broad build errors match baseline. Race compilation still lacks `gcc`.

[Evidence](evidence.json), [metrics](metrics.json), static validation, all initial/final logs and both resource monitors are retained. The optional go-264 adapter remains unmerged pending separate review; its corrected provider and isolated commit are not prerequisites for this kernel. No push, deployment or restart occurred.
