# Experimental tiled CPU convolution

The explicit `WeSpeakerBlockGEMM` mode reduces the measured30-second Go diarization job from49.235 to40.285seconds in a balanced ABBA comparison (1.222×). Segmentation, full/exclusive turns and DER still match the pinned reference. Existing scalar/SIMD modes and defaults are unchanged. Tight intermediate comparisons regress, and full-size scalar fallback times out; the candidate is not production-qualified.

## Profile and candidate

The original complete-job CPU profile records48.24seconds of samples over48.15seconds elapsed. Embedding accounts for81.20% cumulative CPU, segmentation18.06%. `dotRowsx4Asm` accounts for38.79% flat samples and `weSpeakerBlockConv`27.22% flat /69.65% cumulative. Profile percentages are sampled CPU attribution, not hardware-counter or energy attribution.

The original convolution constructs a patch and runs GEMV for each spatial position. The candidate packs64 spatial positions into reduction-major `[K,N]`, uses the existing Plan9 SGEMM NN kernel, and scatters `[channels,N]` into CHW output. Spatial offsets are computed once per kernel tap and shared across input channels. Contiguous packing replaced an initial strided implementation after a diagnostic profile showed packing overhead; the first experiment is retained.

Packed scratch is at most576KiB for256input channels and a3×3kernel, plus64KiB projection scratch and4.5KiB offsets on64-bit hosts. No full-window im2col, weight repacking, worker pool or model-global mutable state is added. Cancellation is checked during channel packing and between bounded tile dispatches.

The new reusable `FMAMatrixF32Checked` overwrites dense C with A×B. It checks exact lengths, finite A/B, non-overlapping output, and bounds M/N1..256/K1..4096 before writes. Read-only inputs may overlap. It allocates nothing and retains no state. Finite arithmetic overflow is allowed; model callers reject nonfinite outputs.

On amd64 it uses the existing immutable AVX2/FMA probe, then one assembly wrapper checks MXCSR rounding mode/DAZ/FTZ, clears C and dispatches SGEMM with alpha1. Exception masks are caller responsibility and are not checked or changed. Other architectures or disabled ISA use `FMA32Scalar` in ascending-K order. The SGEMM kernel is reused, not rewritten. Default GEMV uses a different horizontal reduction order; baseline and candidate are not bit-exact to each other.

## Balanced measurements

The neighbouring workstream explicitly confirmed its hold before the timing screen and full ABBA run. CPU process snapshots and one-second memory/swap samples are retained. This is coordinated shared-host testing, not an enforced host-wide isolation guarantee.

### Resident embedding trunk

Input is the pinned five-second public fixture's reference Fbank, with the same resident trained weights. Each run uses two alternating ABBA/BAAB blocks: four samples per mode,24timed evaluations across three runs. Timings include packing, validation, output allocation, CNN and BatchNorm; they exclude model loading, PCM frontend and pooling. Every timed output is checked bit-for-bit against a warm output of the same mode outside timing.

| Run | Baseline median | Candidate median | Speedup |
|---|---:|---:|---:|
| 1 | 0.939714s | 0.758909s | 1.23824× |
| 2 | 0.936387s | 0.763859s | 1.22586× |
| 3 | 0.939221s | 0.758379s | 1.23846× |

The maximum baseline/candidate trunk difference is5.781650543212891e-6. This screen does not establish arbitrary-shape or hardware-wide speedup.

### Complete30-second job

Separate fresh test processes run in A-B-B-A order, after the exploratory candidate check:

| Order | Mode | Test duration |
|---|---|---:|
| A1 | Existing SIMD/GEMV | 49.14s |
| B1 | Tiled GEMM | 40.31s |
| B2 | Tiled GEMM | 40.26s |
| A2 | Existing SIMD/GEMV | 49.33s |

Two-sample medians:49.235s versus40.285s, ratio1.222167. These durations include test setup, hash verification, model loading and complete inference; JSON output writing is inside the test. They are not pure neural-kernel timings. Both baseline results are byte-identical to one another; both candidate results are byte-identical to one another. All four results pass the saved fresh-reference mask/embedding/turn/DER gate.

During the full ABBA window,178samples recorded minimum MemAvailable28,097,308KiB (about26.80GiB). Swap-in36334 and swap-out635938 stayed unchanged. The trunk screen's27samples also had unchanged counters. Sampling cannot exclude sub-second pressure or establish thermal/power attribution.

Forty seconds for30seconds of audio remains slower than real time and far below the planned combined-job target. No default promotion is made.

## Numerical qualification

- Matrix tests establish bit-exact ascending-K scalar/FMA output across odd/even rows, column tails around8/32/64, reduction lengths through4096, unaligned destinations, admitted edges and model K2304.
- Guard-page tests cover A/B/C at both accessible-page boundaries, rows1/2/3, columns1..65, reductions1/3/7. MXCSR rejection leaves C untouched; signed zero, subnormal and overflow outputs match the exact scalar path. Bad lengths, NaN/Inf, output overlap, dimensions and mutable legacy dispatch flags are tested. Allocation count is zero.
- Tiled convolution tests independently enumerate kernel/padding/stride/channel coordinates and compare exact serial FMA order. The three reduced-width full-depth source oracles pass all17trunk boundaries and pooled embeddings. Cancellation tests cover the new mode through the PCM embedding wrapper.
- Three trained embedding runs:48endpoint comparisons pass the unchanged2e-4gate, maximum error2.771615982055664e-6. However only264/564total strict comparisons pass, compared with300/564in the original mode. Thus300strict comparisons fail for the candidate versus264for baseline. Tight trunk errors regress; Fbank and soft-mask support failures remain. The candidate's strict target fails all four cases.
- Complete candidate diarization keeps37training rows,2clusters,13full/12exclusive turns and84ambiguous frames. All37,107segmentation values match the pinned reference; embedding maximum error3.4570693969726562e-6. Full/exclusive turns match within1e-12seconds and full DER remains5.207392%/1.317798% at0/.25collars, zero delta. This remains one public source with explicit lowest-index ties; all prior strict segmentation/tie failures are retained.

The forced AVX2/FMA-off full embedding run passed silence, broadband and the one-second public case, then hit the180-second deadline in the five-second public case. That failed log is retained. Small-kernel and reduced-model fallback checks pass; full-size fallback latency/qualification does not. No automatic fallback retry with a longer deadline was used.

## Verification and review

- `make speech-community-gemm-check speech-foundations-check speech-media-integration`: pass. The GEMM target runs enabled/disabled ISA paths for bounded kernel/model tests.
- Community-1 plus SIMD regressions:644passing events /287top-level passes. Eight explicit diagnostic/trained/timing test skips plus the no-tests SIMD package skip are recorded.
- Thirty shuffled new matrix/tile/cancellation tests:210passing events, zero skips/failures.
- Affected Community-1 vet passes; SIMD retains exactly three pre-existing q8dot ABI warnings, and other SIMD vet analysers pass. New assembly ABI checks report no warning.
- Community-1 and SIMD arm64 test binaries cross-build; not executed on arm64. Broad/backend compilation retains baseline errors; race compilation lacks `gcc`.
- Assembly listing confirms the181-byte guard/zero/forward wrapper. A scoped delegated review checked ABI0 arguments, outgoing stack slots, prewrite safety, immutable ISA admission, packing/scatter and reduction order; it found no concrete bug. The documented exception-mask scope was corrected after review. This is not a full model/hardware audit.

## Reproduce

```sh
GOMAXPROCS=2 CGO_ENABLED=0 make speech-community-gemm-check
export GO_PHERENCE_COMMUNITY1_EMBEDDING_DIR=/path/to/pinned/embedding-cache
GOMAXPROCS=2 CGO_ENABLED=0 make speech-community-gemm-timing
GO_PHERENCE_TEST_COMMUNITY1_GEMM=1 GOMAXPROCS=2 CGO_ENABLED=0 \
  make speech-community-embedding-check
# Requires the same segmentation/embedding/PLDA/public-PCM variables as the prior milestone:
GO_PHERENCE_TEST_COMMUNITY1_GEMM=1 GOMAXPROCS=2 CGO_ENABLED=0 \
  make speech-community-diarization-lowest-ties
```

The test-only GEMM environment switch does not select a runtime default; public callers choose the explicit enum. [Evidence](evidence.json) retains original/candidate profiles, timings, resource monitoring, strict regressions and timeout. Model weights remain outside source control. Services remain inactive; no GPU use, go-264 merge, push, deployment or restart. Compute window: `whisper-community-cpu-1035`.
