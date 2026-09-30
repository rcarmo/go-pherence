# Nemotron 3 Diarization roadmap

Go-pherence runs the pinned `nvidia/Nemotron-3-Diarization` checkpoint
through a bounded native PCM-to-logits-to-segments streaming path. SIMD has
model-oracle parity on JFK, tiled JFK, a podcast crop and a synthetic
speaker transition. PTX and Vulkan can optionally run one stacking
projection inside the CPU request. Neither GPU path runs the full audio
tower, speaker cache and head. See [the bounded reference record](../validation/nemotron-speech-reference-2026-09-28.md).

## Hybrid GPU projection requests

`PCMStreamingRequest.Projector` optionally selects a per-request
`DeviceStackingProjector{Backend: "ptx"}` or `"vulkan"`. It keeps the
released `[512,1024]` stacking weight and bounded 64-row device buffers
resident until `Close`. Each complete, padded or first-window terminal-correction stack group
uploads its input, dispatches a fixed 64-row projection with zeroed
tail, then downloads owned output. The PCM frontend, 31-layer audio tower, speaker
cache, head and segment extraction remain on CPU. Callers must close
the projector on success or failure. A device projection failure closes
the consumed stacking stream; its eight-row group cannot be replayed.

On the i7-12700/RTX 3060 with `GOMAXPROCS=4`, the synthetic 31-second
JFK→podcast request took `11.92 s` on SIMD, `12.24 s` with PTX and
`12.25 s` with Vulkan (one sample each, excluding model load; including
projector preparation, transfers, dispatches and teardown). Both hybrids
issued nine stack dispatches, matched all 3,099 pinned logit rows with
zero outliers under `3e-4 + 2e-5*abs(reference)`, and returned the same
four model-assigned spans. Both also passed the 11-second JFK and
20-second podcast references. A 16,640-sample boundary fixture checks
the masked terminal-correction projection on GPU: both backends dispatched
three stack groups and returned all 103 pinned logits rows with zero
outliers. On 100-second tiled JFK, PTX took
`90.55 s` and Vulkan `90.67 s`, each returning 9,999 logits rows and
29 pinned spans with zero outliers. These single-host results show no
whole-request GPU speedup; the GPU does not run the full inference
pipeline. They do not establish labelled DER or sustained throughput.
Run the opt-in segment parity test with
`GO_PHERENCE_NEMOTRON_DIARIZATION_STACK_PROJECTOR=ptx` or `vulkan`
and matching model, logits and segments references.

A further PTX trial moved the 30 remaining audio layers' MLP GEMMs onto
resident model-owned weights while leaving attention, layer normalisation,
cache and head on CPU. On the same i7-12700/RTX 3060, the mixed JFK→podcast
request still matched 3,099 reference logits rows and four spans, but took
`14.27 s` versus a `12.71 s` CPU sample. A fused variant retained the
2,048-wide intermediate across fc1, GPU row bias, exact-erf GELU and fc2;
it also matched pinned output yet took `22.18 s`. Both changes were
reverted. Per-layer activation uploads and downloads defeat those hybrids.
A useful tower GPU implementation must keep activations resident across
attention, MLP, normalisation and the cache/head boundary; the isolated
MLP timings supply no speedup claim.

A plan-compatible Vulkan time-major sequence RoPE operator now rotates
`[rows,heads,headDim]` Q/K in one dispatch while keeping activations in
a tensor arena. The checked F32 shader owns both elements of each pair,
rejects frequency/input overlap and bounds rows, heads and storage before
dispatch. A 17-row native plan run matched the independent CPU operation
(maximum absolute error `1.19e-7`). On the RTX 3060, released-weight
138-row layer-1 Q and K fixtures matched the pinned PyTorch post-RoPE
reference with maximum errors `4.02e-6` and `3.40e-6`, means
`1.34e-7` and `1.36e-7`, and zero outliers under the existing
`3e-4 + 2e-5*abs(reference)` gate. The new SPIR-V passes `spirv-val`,
embedded contract and byte/normalised offline rebuild checks. The
repository-wide offline shader rebuild check still fails on 13 older
shaders with the installed compiler; the new RoPE shader passes.
This prepares one operation for a resident tower. It does not run a
complete GPU attention layer, speaker cache or PCM request.

A complete **single audio layer** now runs in a fixed-row resident Vulkan
plan: LayerNorm→Q/K/V projections→time-major RoPE→non-causal eight-head
attention→output projection/residual→LayerNorm→fc1→erf GELU→fc2/residual.
The reusable plan uploads the complete prepared input once and downloads
the output once; its intermediate F32 tensors and released layer weights
remain in a bounded arena. At 16 and 138 rows, native RTX 3060 outputs
matched independent PyTorch layer-1 fixtures with maximum/mean absolute
errors `2.67e-5`/`5.83e-7` and `2.29e-5`/`5.12e-7`, respectively,
with zero values outside `3e-4 + 2e-5*abs(reference)`. Device memory
usage returned to baseline after `Close`. This tests an unmasked layer
starting at position zero; it does not exercise the streaming cache or head.

An opt-in 31-layer hybrid now runs the first audio layer on CPU, the other
30 layers and final normalisation in fixed-row Vulkan plans, and the head
and speaker cache on CPU. Intermediate layer activations stay on Vulkan;
each plan fences before the next layer, while a change in prepared row count
closes and rebuilds the resident tower. A 138-row pinned PyTorch fixture
passed at layers 2, 7, 15, 23 and 30 and the final norm. Layer 30 had
maximum/mean absolute error `6.71e-4`/`1.49e-5`; after final norm the
errors were `5.72e-6`/`3.74e-7`. No values exceeded
`3e-4 + 2e-5*abs(reference)`. Eleven streaming windows passed pinned
logits at steps 0, 1, 2, 5 and 10. The opt-in CLI uses `-task diarization
-backend vulkan -vulkan-tower`, with explicit device failure and teardown.

On the i7-12700/RTX 3060 with `GOMAXPROCS=4`, the opt-in 11-second JFK
request took `9.45 s` versus `2.03 s` on SIMD (one CLI sample each;
excludes WAV/checkpoint load, includes setup, transfers and teardown).
The 100-second tiled JFK segment test took `108.31 s` with Vulkan stacking
and the resident tower versus `86.79 s` on SIMD (one test sample each).
The independent PyTorch references accepted 1,099 and 9,999 logits rows,
respectively, with zero outliers and the same model-assigned spans. A
separate 100-second tower-only projection-disabled run passed all 9,999
rows; the original fixed 1 MiB final-norm arena failed as the cache grew
and was replaced with a row-bounded allocation. A five-iteration fixed-row tower benchmark, including layer 0 and transfers,
measured `79–84 ms` per 138-row Vulkan run versus `143–198 ms` on SIMD
(two samples per backend). Each Vulkan setup took `0.43 s` and teardown
about `0.11 s`. Prepared streaming row counts change as cache grows, so
repeated plan construction costs more than the resident-layer gain in these
complete requests. A Vulkan prefix-row view and atomic plan `Rebind` now
provide a bounded way to keep arena storage, kernels and plan descriptors
while changing the exact live row count. An RTX 3060 test rebound RoPE and
non-causal eight-head attention at 17, 5 and 11 rows using one plan and
arena; independent float64 attention estimates had maximum errors
`4.21e-8` or lower, with no padded key entering softmax. Offline rollback,
cancellation, owner-close, resource and `-race` checks passed. The 30-layer streaming tower now reuses one maximum-row allocation, all 30
sets of weights/kernels and their plan descriptors per request, rebinding
exact prefix rows for each streaming window. On the i7-12700/RTX 3060 at `GOMAXPROCS=4`, the 11-second JFK request
with Vulkan stacking took `2.41 s` and `1.70 s` in two samples (versus
`1.89 s` SIMD on the same revision); the 100-second tiled JFK request
took `35.74 s` and `30.31 s` (versus `75.13 s` SIMD on the same revision).
The previous Vulkan tower rebuilt each shape and took `9.32 s`/`108.31 s`
for those requests. All timings exclude model/WAV loading and include
setup, transfers and teardown. These bounded samples show a 100-second
request-level hybrid GPU speedup on this host; 11-second timing overlaps
SIMD, and sustained throughput or other hardware is untested.
Both matched all 1,099/9,999 independent PyTorch logit rows with zero
outliers and the same three/29 model-assigned spans. The synthetic 31-second
JFK→podcast request took `6.10 s` on this Vulkan hybrid versus `12.25 s`
on SIMD on the same revision, matching all 3,099 pinned logit rows and four
spans (including the model-assigned second speaker); one sample each.
The 11-window pinned
fixtures, native `-race`, cancelled-run, repeated-run and cleanup tests pass.
A same-revision 100-second PTX stacking-projection hybrid took `75.69 s`
with 9,999 pinned logits rows, 29 matching spans and zero outliers. SIMD
on that revision took `75.13 s`; PTX still runs all 31 audio layers, head
and cache on CPU. The Vulkan request still runs its frontend, first layer,
head and speaker cache on CPU. Labelled diarization quality and complete
GPU-request performance are open.

The affine F32 PTX LayerNorm operator now uses separate mean and centred-variance
reductions, then applies per-column gamma and beta on resident buffers. The
released layer-1 PyTorch fixture covers 16 and 138 rows: maximum absolute
error was `7.60e-7` and `9.54e-7`, mean absolute error was `4.47e-8` and
`4.44e-8`, and both had zero outliers at the existing
`3e-4 + 2e-5*abs(reference)` gate. The 16-row normal was generated by
`scripts/nemotron_diarization_qkv_fixture.py` in the pinned CPU Transformers
5.18 development environment. A separate 10,000–30,000-offset F32 test had
maximum error `4.38e-4` against a centred F64 reference at its `2e-3` gate;
odd widths, in-place output and malformed buffers also passed. These are
operator tests on RTX 3060. The resident PTX path and its later request
measurements are documented below.

A separate exact-key operator check now composes the existing PTX sequence
RoPE and `attention_full_online` kernels with the released layer-1 Q/K/V and
a CPU output projection. At 16 and 138 rows on RTX 3060, attention matched
the independent PyTorch fixture with maximum absolute errors `3.67e-5` and
`2.86e-5`, mean errors `2.47e-7` and `1.72e-7`, and zero outliers at the
existing gate; three repeat runs passed. The online kernel was selected
explicitly for this test. The initial shared-score `attention_full` gave
variable errors at 138 rows, including outliers; its later barrier fix and
resident-request evidence are documented below. Sequence RoPE checks position addition,
u32 element indices and buffer byte extents before launching. This checks an
isolated attention operator, not a resident PTX layer or complete request.

The existing PTX erf-GELU now has a checked, GPU-resident F32 buffer entry
point that reports errors without CPU fallback. Its 20,001-value `[-10,10]`
probe on RTX 3060 differed from an independent F64 `math.Erf` reference by at
most `4.77e-7`, with mean absolute error `4.67e-8` and zero values outside
the existing `3e-6` operator gate. Malformed buffer and dimension tests pass.
A fixture-gated resident layer-1 MLP probe keeps the second affine norm,
transposed fc1/fc2 weights, row biases, erf-GELU and residual addition on PTX
until final download. Its CPU-composed attention residual feeds the probe.
The 16/138-row complete-layer outputs match independent PyTorch fixtures with
maximum absolute error `2.29e-5` at both lengths, mean errors `5.65e-7` and
`5.14e-7`, and zero outliers at the existing gate. The model-owned residual
and GPU copy remain unchanged. The production `PTXAudioLayer` now composes
Q/K/V projection, sequence RoPE, explicitly selected shared-score attention
(after its max/sum barrier fix), output projection, both affine norms and the complete MLP without downloading
intermediate activations. It retains transposed weights and frequencies,
accepts exact 16/138-row unmasked windows, serialises forward and close, and
returns errors without CPU fallback. On RTX 3060 its output matched the
independent complete-layer PyTorch fixtures at 16 and 138 rows (maximum
absolute error `2.67e-5` and `2.29e-5`, mean `5.73e-7` and `5.11e-7`, zero
outliers). Repeated 16-row, input-ownership, malformed-input, shared-instance
concurrency and direct-buffer no-transfer/scratch-balance gates passed.

The opt-in `PTXAudioTower` now retains all 30 remaining layers and final
normalisation weights on CUDA. Layer 0 and the request frontend/head/cache
stay on CPU. `ForwardRows` executes exact unmasked prefixes against one
maximum-row weight allocation; the 138-row final output matches the released
PyTorch fixture (maximum absolute error `6.20e-6`, mean `3.73e-7`, zero
outliers at the existing gate). The 16→138→16 and 13-row sequences match CPU
composition within `5.25e-6` and `4.77e-6` maximum error respectively; those
short-row comparisons are not independent PyTorch final-norm fixtures. Direct
stats count one host-to-device input and one device-to-host output transfer per
tower call with balanced scratch allocations. Repeated race, shared-instance,
cancellation-before-run and close gates pass. Streaming integration uses one
per-request tower and the same exact row count through its speaker-cache
windows. A missing shared-memory barrier in the online attention kernel had
caused intermittent drift at 103 keys on repeated identical Q/K/V inputs;
a regression probe and repeated 103-row streaming windows now pass after the
barrier, without changing numerical tolerances. Pinned 100-second PTX-tower
PCM→logits→segments on i7-12700/RTX 3060 at `GOMAXPROCS=4` took `82.57 s`
inside the complete-request test, returned 9,999 logit rows and the same 29
model-assigned spans, with maximum/mean logit error `6.48e-5`/`5.26e-6` and
zero outliers. This sample is slower than the earlier 75.69-second PTX
projection-only request. The CLI exposes `-backend ptx -ptx-tower` as an
explicit hybrid; repeat CLI timing, cancellation soak, labelled quality and
sustained throughput remain open.

A bounded tower-owned scratch set now replaces nine alloc/free pairs per
layer and defers one CUDA drain until the 30-layer call completes. On the
same i7-12700/RTX 3060 (`GOMAXPROCS=4`, five warm repetitions per case,
two runs), exact 103-row tower calls changed from `75–76 ms` and about
`7,099` Go allocations to `71–74 ms` and about `3,015` allocations. The
138-row calls changed from `115–119 ms` to about `107 ms`; setup remained
about `0.5 s`. The independent 138-row PyTorch fixture, repeated 103-row
stream windows and race gates passed unchanged. One pinned 100-second
PCM→logits→segments sample took `80.81 s`, again with 9,999 rows, 29 spans
and zero outliers. It is still slower than the prior projection-only PTX
sample; the tower benchmark is not a complete-request speed claim.

The explicit `SgemmReg2` candidate now serves only the PTX diarization
layer's six projection calls; general SGEMM dispatch remains on its oracle.
An independent F64 matrix-product test at 13/103/138-row F32 shapes found
zero outliers for oracle, reg2 and the repaired skinny candidate. The skinny
kernel previously wrote A shared-memory rows from lanes 16–31 beyond their
16-element fragments; its store is now lane-guarded. At 103 rows, the reg2
candidate took `71–73 µs` for 512×512, `255–256 µs` for fc1 and about
`277 µs` for fc2 versus oracle `97 µs`, `361–363 µs` and `371 µs` in the
bounded resident-kernel probe. The PTX tower at 103 rows took `64–68 ms`
versus the prior shared-scratch `71–74 ms` in two five-call samples. One
pinned 100-second PTX-tower request took `77.73 s` with 9,999 matching
logits rows, the same 29 spans and zero outliers. This is still slower than
the previous `75.69 s` projection-only PTX sample; unrelated SGEMM callers
retain oracle dispatch.

A bounded 103-row stage probe measured CPU layer 0 at `4.89–7.28 ms`,
one H2D transfer at `25–34 µs`, the 30-layer PTX tower/final norm at
`57.70–59.15 ms`, one D2H transfer at `121–136 µs`, and the CPU head at
`4.67–5.30 ms` (three samples; diagnostic wall times). The existing
shared-score `attention_full` had shown variable 138-row outliers; adding
a barrier before its shared max/sum buffer reuse made 20 repeated 103-key
launches agree with the separately qualified online kernel at the existing
`2e-5` operator gate. With the shared-score path selected only for PTX
diarization, 16/138-row layer and tower fixtures and repeated streaming
windows pass with zero outliers. At 103 rows, five-call warmed tower samples
fell from `64–68 ms` to `33–36 ms`; the pinned 100-second request took
`23.71 s` (test) and `23.60 s` (CLI), returned 9,999 matching logits rows
and the same 29 spans with zero outliers. This remains a hybrid request on
one i7-12700/RTX 3060 host; no full-GPU or labelled DER claim follows.

The CPU head now avoids materialising a channel-major convolution copy and
a separate pre-activation upsampled tensor on production logits calls;
pinned intermediate tests retain both diagnostic stages. A 138-row released
head fixture passes all projected, convolved, upsampled and logits gates;
production logits match the diagnostic path value for value. A 30-second
PCM benchmark at `GOMAXPROCS=4` reduced allocation from about `7.236 GB/op`
to `7.133 GB/op` (two samples each); observed timings overlap (`13.70–15.85 s`
before and `13.19–15.29 s` after), so this is an allocation reduction,
not a speedup. The pinned 11-second and 100-second streaming logits and
spans still pass with zero outliers. The CPU audio tower also reuses each
layer's owned normalisation buffer as the fc2 destination after fc1 consumes
it; the accumulating GEMM clears that buffer before writing. The 16/138-row
layer and 138-row tower PyTorch fixtures pass under `-race`. Two 30-second
request samples allocated about `6.603 GB/op` after the reuse, down from
`7.133 GB/op` after the head change. The SIMD attention path then reuses
each layer's owned Q buffer for mixed values once a row's Q scores are
complete. Scalar and vector 16/138-row attention fixtures, the 138-row
tower and the 11-second streaming logits/spans pass; the 30-second request
allocated `6.090 GB/op` in two samples versus `6.603 GB/op` before Q reuse.
The SIMD attention tile now reuses its copied Q-head buffer for the
post-softmax value product, once scores have consumed Q. One 30-second
request sample allocated `6.023 GB/op`. Layer 0 now applies the same owned
Q/head-tile reuse; a separate 30-second request sample allocated
`6.004 GB/op`. The pinned layer-0 and layer-1 attention/layer fixtures,
138-row tower and 11-second stream still pass under their numeric gates.
The 100-second tiled JFK request passed after the layer-0 change: 9,999
pinned logits rows, 29 spans, zero outliers (`73.93 s` single request). Timing samples remain noisy and supply no
speed claim. The independent
100-second reference accepted all 9,999 logits rows and 29 spans after
Q reuse (single `GOMAXPROCS=4` request: `74.83 s`, zero logit outliers).
It passed again after the head-tile reuse (`76.20 s`, 9,999 rows, 29
spans, zero logit outliers; one request sample). An isolated packed/parallel head GEMM
trial looked faster at fixed shapes but failed to improve the complete
30-second request, so that dispatch change was reverted.

## Pinned source and scope

- Model: [NVIDIA Nemotron 3 Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization/tree/a435e9867d79e789e90053f9b6d6834053af564a), revision `a435e9867d79e789e90053f9b6d6834053af564a`.
- Published [configuration](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/a435e9867d79e789e90053f9b6d6834053af564a/config.json): `Nemotron3DiarizationForAudioFrameClassification`, 31 audio-transformer layers, width 512, 128 mel bins, factor-eight subsampling, and an eight-speaker head. The configuration also defines chunk/FIFO and speaker-cache state; their numerical and temporal behaviour has not been verified here.
- The [model card](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/a435e9867d79e789e90053f9b6d6834053af564a/README.md) describes offline and streaming diarisation for up to eight speakers. The Hub records `openmdw-1.1` licence metadata. Its OpenMDW-1.1 terms require retaining the licence and applicable copyright/origin notices on redistribution and leave third-party rights/permissions to the user; generated outputs have no licence-imposed obligations. Review deployment terms before checkpoint use.

The existing [Whisper and Community-1 workflow](speech-integration.md) keeps
plain transcript output independent of optional speaker attribution. Nemotron
would be a distinct opt-in speaker provider, not a silent replacement for
Community-1. Neither backend has trained-quality acceptance in the integrated
speech job.

## Pinned metadata inspection

At revision `a435e9867d79e789e90053f9b6d6834053af564a`, the public file list includes `config.json`, `processor_config.json`, `README.md`, an ASR integration guide, `.nemo` and safetensor checkpoints. The safetensors checkpoint was later fetched and integrity-checked after user approval; the isolated CPU reference run is recorded separately. The processor uses 16 kHz mono input, pre-emphasis 0.97, 512-point FFT, 400-sample window and 160-sample hop with 128 mel features. A factor-eight subsampler yields a nominal 80 ms model step; the card describes `(batch, frames, 8)` logits every 10 ms for its exposed frame classification, so timestamp alignment must be verified against an oracle rather than inferred from the subsampling factor alone.

The processor's default `low_latency` mode lists `[9,4]`; `very_low_latency` and `ultra_low_latency` list `[6,2]` and `[3,1]`. Model config has 31 attention layers of width 512, an eight-speaker head, `chunk_length: 340` and `chunk_right_context: 40`. It also distinguishes top-level FIFO/cache settings from a separate `streaming_config` (`fifo_length: 264`, `speaker_cache_length: 264`, update period 222). The card's Transformers streaming example passes `speaker_cache` between chunks and marks the last chunk explicitly. The exact units, mode-dependent lookahead and cache ownership require independent fixtures before a native streaming implementation.

## Proposed gates

1. Freeze the pinned configuration, processor contract, asset inventory,
   licences and file hashes. Check sample rate, feature framing, channel order,
   timestamp origin and maximum-speaker rules.
2. User approval now covers downloading needed Nemotron weights. Produce
   independently generated small fixtures for the frontend, subsampling,
   transformer/head, arrival-order speaker cache, streaming state and offline
   frame-to-turn decoding. Record the oracle software version, input hashes,
   output hashes, units and numerical tolerances.
3. Implement a scalar Go reference with bounded allocations and checked state
   transitions. Promote measured hot operations to checked SIMD backends only
   after operator and whole-recording parity. A Python/NeMo process is an oracle
   option, not the Go inference runtime.
4. Add an explicit speech-job provider selection while preserving plain
   transcript/VTT on diarisation failure. Test overlapping speakers, silence,
   unknown speakers, chunk seams, cancellation/retry, long recordings and
   deterministic turn labelling without reusing Community-1 checkpoints.
5. Measure diarisation error rate on an independently labelled cohort, both
   offline and streaming latency/throughput on each proposed native platform.
   Record speaker-count and recording-length limits. Do not infer accuracy from
   model metadata or synthetic operator tests.

The released checkpoint ran once through the pinned Transformers CPU reference.
The bounded 16-row `model/nemotrondiarization/Layer0QKV` path applies both
`audio_tower.input_layer_norm` and the first layer's `layer_norm1` before Q/K/V.
It matches independent PyTorch fixtures: maximum absolute error was
0.00000382 for Q, 0.00000286 for K and 0.000000834 for V on AVX2/FMA, with
zero values outside the documented absolute/relative tolerance. A one-CPU
microbenchmark of the 16-row operation took 0.396–0.490 ms, 163,840 B and
five allocations across five 20-iteration runs. Loading, frontend, stacking,
RoPE, attention and transfer costs are excluded. The 31-layer encoder, cache,
upsampler and speaker head have no native parity. `Layer0Attention.ForwardOffline`
passes PyTorch parity for an isolated unmasked 16-row bidirectional window:
maximum absolute error `5.72e-6` for the attention projection and `7.63e-6`
for its residual. Its one-CPU, five-by-20-iteration microbenchmark used
294,976 B/10 allocations and 0.877–2.306 ms per window; timings are noisy.
The window does not reproduce the first 16 rows of the full 138-row recording.
The full 138-row first-layer attention also matches the PyTorch reference on
JFK: max error `7.63e-6` for attention and `1.14e-5` after residual. The
processor marks 1100 of 1101 feature frames valid; sampling that mask every
eight frames marks all 138 stacked positions valid. The one-CPU 138-row
benchmark after sharing the owned input-normalisation buffer took 22.4–29.5 ms
and 2,294,336 B/9 allocations in five 10-iteration runs, down from
2,581,056 B/10 allocations. Timing samples overlap, so there is no speed
claim. The bounded complete first audio layer adds `layer_norm2`, the 512→2048→512
GELU MLP and its residual. Independent PyTorch 16- and 138-row outputs pass:
full-context maximum absolute error `1.53e-5`, mean `3.33e-7`, with zero
values outside the calibrated tolerance on AVX2/FMA. The scalar fallback
passes too. One-CPU 138-row samples took 43.1–44.3 ms with 3,998,272 B/12
allocations across five 10-iteration runs. Sharing each position's RoPE
sine/cosine across eight heads kept fixture parity; subsequent five
10-iteration runs took 41.8–43.9 ms for the full layer and 21.6–22.5 ms
for attention alone, with unchanged allocations. Timing samples overlap, so
this does not establish a speedup. Checked SGEMM for the two attention matrix
products passed the independent PyTorch fixture and scalar comparison for
windows of 1, 7, 15, 16, 17, 64, 138 and 376 rows. At 138 rows, three
20-iteration single-CPU samples took 7.03–8.55 ms for attention and
26.75–31.73 ms for the complete layer, versus the prior attention samples
of 21.6–22.5 ms. Scratch increased attention allocations from 2,294,336 B/9
to 2,539,520 B/13 (the complete layer uses 4,243,456 B/16 allocations).
These calls exclude loading, frontend,
stacking and the other 30 layers. The composed 138-row layer-0 output also passes the second layer's
pre-attention normalisation and Q/K/V fixture checks. On AVX2/FMA, maximum
absolute errors are `9.54e-7` for normalisation, `3.34e-6` for Q,
`2.86e-6` for K and `9.54e-7` for V, with zero values outside tolerance.
The second layer's full-window attention and residual now pass separate
16- and 138-row PyTorch fixtures via `Layer1Attention.ForwardOffline`. Both
checked SGEMM and scalar paths have zero values outside
`3e-4 + 2e-5*abs(reference)` and mean error below `2e-6`. The largest
residual error is `3.05e-5` on AVX2/FMA and `6.10e-5` with CPU features
disabled; 16-row and 138-row windows have different
bidirectional contexts. These tests include the composed first layer and exclude loading, frontend
and stacking. `Layer1Complete.ForwardOffline` also composes the second
layer's normalised GELU MLP and passes independent 16- and 138-row output
fixtures: AVX2/FMA maximum absolute error `2.29e-5`, scalar fallback
`3.81e-5`, mean below `1e-6`, and zero per-value outliers. The layer-1 math also loads an indexed layer from the released 31-layer
audio tower. A separately generated full-context 138-row PyTorch fixture
checks the composed first three layers through layer-2 normalisation,
attention, residual and complete GELU MLP: AVX2/FMA maximum output error
`2.29e-5`, scalar fallback `5.34e-5`, mean below `2e-6`, with zero
per-value outliers. `OfflineAudioTower.ForwardOffline` now composes all 31 released audio layers
and the tower's final LayerNorm over the same unmasked 138-row window. Pinned
PyTorch checkpoints after layers 7, 15, 23 and 30, plus the final norm,
pass AVX2/FMA and CPU-features-disabled checks with zero values outside
`3e-4 + 2e-5*abs(reference)`. The raw layer-30 output has AVX2/FMA maximum
error `8.55e-4`, mean `2.08e-5` (scalar maximum `1.10e-3`, mean `2.40e-5`);
its measured mean gate is `3e-5`. The final-normalised output has AVX2/FMA
maximum error `4.77e-6`, mean `5.66e-7` (scalar maximum `1.24e-5`, mean
`6.51e-7`), with a `2e-5` mean gate. These are one-recording, full-context
encoder measurements; they do not qualify arbitrary lengths, masks, chunking,
or concurrent serving. Frontend composition, cache, upsampler and speaker head
have no native end-to-end parity or full-request latency evidence. A separate `OfflineHead.ForwardOffline`
path now matches pinned PyTorch projection, 3-tap subpixel convolution,
upsampling and eight-speaker logits for the 138-row tower output. The isolated
head's logits have AVX2/FMA maximum error `1.53e-5`, mean `7.03e-7`;
scalar maximum `1.91e-5`, mean `1.45e-6`, with zero per-value outliers.
Composing the Go tower and head yields 1104×8 logits before trimming to
valid input frames: AVX2/FMA maximum error `1.91e-5`, mean `3.41e-6`;
scalar maximum `2.67e-5`, mean `3.67e-6`, with zero outliers. These
prepared-stacking windows isolate model numerics.

The shared `loader/audio/NemotronMelStream` frontend processes 100 seconds of
tiled JFK PCM in five-second calls with fixed state and owned feature chunks.
Against full-duration pinned Transformers features, its 10,001 rows have a
maximum error of `3.67e-4`, mean `1.15e-6` and zero tolerance outliers.
`PCMStackingStream` now composes that frontend with the released stacking
projection, emitting owned embeddings from at most five seconds of PCM per
call with at most seven pending mel rows. On the same 100-second audio, its
1,251 embeddings match independent PyTorch output: SIMD maximum error
`2.68e-4`, mean `9.25e-6`; CPU-features-disabled maximum `2.90e-4`, mean
`1.06e-5`, with zero per-value outliers under `3e-4 + 2e-5*abs(reference)`.
Prepared-feature chunk-boundary tests also pass two chunk patterns. The 31-layer
model still rejects offline windows above 376 rows; streaming model scheduling,
speaker-cache policy and masks have not been connected to the PCM stream.

`OfflineRequest.ForwardPCM` now composes the native 16-kHz log-mel frontend, stack projection, 31-layer
tower and head on the pinned 176,000-sample JFK waveform. The owned 1101×8
raw logits match the independent PyTorch request: AVX2/FMA maximum absolute
error `3.25e-5`, mean `2.84e-6`; CPU-features-disabled maximum `6.11e-5`,
mean `5.87e-6`, with zero values outside `3e-4 + 2e-5*abs(reference)`.
The last masked feature row is retained and logits are trimmed from 1104 to
1101 frames, matching the reference request shape. `ExtractSegments` also matches the released processor's default 0.5 activity
threshold, 1100-valid-frame mask and two-decimal timestamps on this request:
speaker 0 spans `[0.28, 2.28]`, `[3.27, 4.56]`, and `[5.36, 10.63]`
seconds. A synthetic overlap/masked-frame check covers ordering and ownership.
This is processor-output parity for one recording, without labelled DER.
The head's subpixel convolution now packs three adjacent projected frames
into checked dense rows, with Conv1d weights reordered once at load time.
On the same i7-12700 with `GOMAXPROCS=1`, five 10-iteration isolated
138-row head samples fell from `133.09–134.77 ms` to `6.36–6.76 ms` per
call. Owned scratch increased from `3,555,328 B/6 allocations` to
`3,874,816 B/7 allocations`. PyTorch stage, composed-logit and PCM-request
fixtures pass SIMD and CPU-features-disabled gates with zero per-value
outliers; the changed reduction order raises the isolated convolution
maximum to `3.63e-5` on AVX2/FMA. These timings exclude loading, frontend,
tower, speaker extraction and transfers; no whole-request speedup has been
measured. The composed CPU path now uses the checked AVX2/FMA `GELUErfF32To` in each
layer's MLP, falling back to scalar exact-erf outside the kernel's range.
Released-weight layer checkpoints, full PCM logits and default-threshold
segments still pass on AVX2/FMA and with CPU features disabled. On the same
i7-12700 with `GOMAXPROCS=1`, five 5-iteration whole-request samples moved
from `856–864 ms` before the GELU change to `565–629 ms` after it, with
`479 allocations` and about `130.37 MB` per request; loading is excluded.
The two sample ranges do not overlap, but a longer controlled latency run
and independent quality evidence are needed before a production speed claim.
A separate `SpeakerFIFO` now matches the pinned streaming cache's
pre-compression embedding transitions for three prepared low-latency chunks
of nine current and four lookahead encoder frames: returned input lengths
13, 22 and 31; retained FIFO lengths 9, 18 and 27. Lookahead frames are
not retained. The released streaming cache uses a 264-frame FIFO and moves
at least 222 frames into a speaker cache on overflow; the separate offline
configuration uses a 40-frame FIFO and 300-frame update period. The Go
pre-compression slice rejects overflow without changing state. `PoolSpeakerProbabilities` also matches the reference's sigmoid and
8-logit average pool for all three prepared steps, with one masked final
encoder row: maximum absolute error `5.97e-8`, mean below `1.3e-8` and
zero values outside `2e-6`. It rejects non-finite logits even on masked
rows and preserves caller input. A separate uncompressed speaker-cache transition now passes pinned steps
at the 264-frame FIFO boundary: the 237-frame append reaches capacity
without a move; the next nine-frame append moves 222 oldest frames into
speaker cache and leaves 51 in FIFO. `SpeakerCacheUncompressed` matches
prepared input order, FIFO state and speaker embeddings; pooled speaker
probabilities differ by at most `5.97e-8`. Lookahead frames are excluded,
and an update requiring score-based compression rejects without changing
state. `SpeakerFrameScores` now matches the released pre-compression scoring rule
on two deterministic 300-row probability fixtures, including speech
probabilities above 0.5, a positive-score branch, and conditional `-Inf`
masking. Finite score maximum absolute error is `9.54e-7`; both fixtures
match every masked position. `SpeakerCompressor.Compress` now checks the released recency boost, strong
and weak top-k boosts, speaker-major ordering, eight learned silence slots
and sentinel fallback on two independent 486-frame probability fixtures.
Prepared pattern and sweep cases match PyTorch selected embeddings and
probabilities with zero per-value differences; the pattern has 70 silence
or sentinel slots and the sweep has eight. The operator leaves inputs
unchanged and does not mutate streaming state. `SpeakerCache.Update` now composes FIFO transfer, stored-versus-reestimated
probabilities and top-k compression on seven deterministic prepared steps.
At step 5 the speaker cache compresses from 444 to 264 frames, retaining 51
FIFO frames; step 6 prepends that compressed order and retains 60 FIFO
frames. Pinned PyTorch embeddings and FIFO state agree, with speaker
probability maximum error `8.95e-8`. Repeating modulo logits originally
created equal-score top-k cutoff ties; the compression fixture now gives
frames distinct scores because PyTorch does not specify which tied frame
survives. `StreamingWindow.ForwardPrepared` now composes the released 31-layer tower,
upsampler/head and owned speaker cache for two consecutive 9+4 prepared
JFK stacking chunks. The second window prepends nine cached frames; both
full-context logit tensors and FIFO states match pinned PyTorch fixtures.
AVX2/FMA maximum logit error is `1.15e-5`, mean below `1.7e-6`;
CPU-features-disabled maximum is `1.91e-5`, with zero values outside
`3e-4 + 2e-5*abs(reference)`. The same prepared JFK sequence now runs 11 consecutive 9+4 windows,
retaining up to 99 FIFO frames. Saved PyTorch checkpoints at steps 0, 1,
2, 5 and 10 match full-context logits, prepared input order and FIFO state;
AVX2/FMA maximum logit error is `1.91e-5`, CPU-features-disabled maximum
`3.44e-5`, with zero per-value outliers. Current-frame logits are selected
from the cached offset at eight output frames per encoder row. This path
admits only fully valid prepared embeddings. Its qualified maximum is now
541 rows (264 speaker + 264 FIFO + 9 current + 4 lookahead). The separate
offline PCM request still has its 376-row limit.
A separate model-level boundary test seeds the reference's prepared
264-frame FIFO and runs two 9+4 JFK windows. The first moves 222 frames
into the speaker cache and leaves 51 in FIFO; the next uses both contexts
and leaves 60 FIFO frames. Full-context logits, speaker embeddings and
FIFO state match pinned PyTorch fixtures. AVX2/FMA logit maximum error is
`4.58e-5`, scalar `5.73e-5`, mean below `6.7e-6`, with zero per-value
outliers; speaker-probability maximum error is below `1.7e-10`. These are
prepared inputs and do not cover score-based compression during model
inference. A separate test seeds the independently checked synthetic
step-5 compressed cache (264 speaker + 51 FIFO frames), then runs two
prepared 9+4 JFK windows with 328 and 337 total model rows. Full-context
logits, input order, stored speaker embeddings/probabilities, FIFO state
and compressed flag match pinned PyTorch. SIMD maximum logit error is
`1.91e-5`, scalar `4.01e-5`, with zero per-value outliers and mean below
`7.9e-6`. This is post-compression inference on a seeded cache, **not**
a continuous model run through the compression transition. A separate
seeded maximum-cache case now processes a 541-row prepared window and
its 328-row successor. The first update compresses the speaker cache;
PyTorch full-context logits, speaker embeddings, FIFO order and compressed
state match on AVX2/FMA and the CPU-features-disabled path. Per-value
logit outliers are zero under `3e-4 + 2e-5*abs(reference)`; maximum and
mean errors are `5.68e-5`/`4.63e-6` on SIMD and `1.34e-4`/`1.16e-5`
on scalar. Pooled speaker-probability maxima are `8.59e-6` and
`1.91e-5`; selected probabilities have maxima `7.34e-6` and
`1.69e-5`, with unchanged speaker/FIFO selection. The fixture seeds
264 compressed speaker rows with distinct stored probabilities and
264 FIFO embeddings from the pinned 100-second JFK projection. The
compressor previously ranked boosts across all speakers; the reference
ranks each speaker's frames separately. The corrected operator retains
its earlier pattern/sweep and seven-step parity fixtures. This case qualifies inference across a **seeded** compression transition.

`PCMStreamingRequest` adds low-latency PCM scheduling with nine committed
encoder rows and four lookahead rows. It retains at most 1,480 PCM samples,
keeps projected context bounded, and scores the last chunk without lookahead.
The first 104-frame chunk followed by a 103-frame terminal output needs two
projections of its last group: the first pass uses the lookahead feature; the
terminal pass masks its 104th feature. The last uncentred chunk excludes
right-padded mel frames, and the scheduler trims its output accordingly.

`nemotron_diarization_pcm_stream_fixture.py` runs the pinned Transformers
model on the complete 11-second JFK recording and a 100-second tiled JFK
recording, carrying its speaker cache between chunks. The Go request returned
1,099 and 9,999 logits rows respectively. Against independent PyTorch logits,
the SIMD path had maximum/mean absolute errors of `3.43e-5`/`2.95e-6`
(11 s) and `6.48e-5`/`4.70e-6` (100 s), with no values outside
`3e-4 + 2e-5*abs(reference)`. The 11-second `GODEBUG=cpu.all=off` run
had maximum/mean errors of `5.34e-5`/`6.44e-6`, also with zero outliers.
Boundary tests additionally compared PCM lengths 160, 200, 11,520, 16,639,
16,640, 16,680, 17,040 and 28,000 samples with PyTorch. The nonfatal
Transformers `image_like_kwargs` diagnostic did not prevent the reference
from producing logits. A full 100-second scalar check has not run.

This qualifies bounded continuous **SIMD diarisation logits**, including
cache compression exercised by the 100-second stream. `SegmentStream`
converts successive committed logits into completed spans using only eight
open-speaker starts. A released-weight 11-second JFK PCM stream with 7,979-
sample calls yields 1,099 logits rows and three spans matching the pinned
PyTorch **streaming** `extract_speaker_dict`: speaker 0 at 0.31–2.25,
3.29–4.53 and 5.40–10.63 seconds. SIMD logits had maximum absolute error
`3.43e-5`, mean `2.94e-6`, zero values outside
`3e-4 + 2e-5*abs(reference)`. The separate offline processor fixture
returns 0.28–2.28, 3.27–4.56 and 5.36–10.63 seconds. Comparing streaming
spans against that offline fixture failed, as the inference contexts differ.
The pinned 100-second tiled-JFK streaming reference also supplies 9,999
logits rows and 29 segments. An earlier native 7,979-sample-call test and a
five-second-call retry each exceeded five minutes without finishing. A
30-second profile identified unblocked `SgemmNT` in the 31-layer tower and
attention scores as the dominant cost. `DenseNTTo` now permits contiguous
bounded windows through 576 rows, and `Layer1Attention` sends its score
matrix through that checked blocked dispatch. Both 11-second and 100-second
SIMD parity pass with this change: the 100-second native PCM→logits→segments
run completed in **246.42 seconds** (five-second PCM calls), matching all
29 pinned streaming spans. Logits had maximum absolute error `6.48e-5`,
mean `4.66e-6`, and zero values outside
`3e-4 + 2e-5*abs(reference)`. A single 30-second request benchmark moved
from 45.02 seconds/7.742 GB to 36.49 seconds/7.742 GB; these single samples
do not establish sustained speed. The 100-second end-to-end path now
finishes, but 246 seconds is slower than real time and remains an
optimisation target. Immutable Q/K/V, attention-output and two MLP weight
matrices in all 31 layers now use model-load-time 16-column SIMD packing;
scalar execution keeps the original released weights. Eight bounded GEMM
shapes showed prepacked multiplication faster than blocked multiplication
in two diagnostic samples apiece, with zero timed allocations. Two
single-iteration 30-second full-request samples fell from `36.49 s` on
the blocked path to `23.62–23.71 s` with prepacking (both still around
7.742 GB/request); the sample size is small. The native 100-second
PCM→logits→segments run then completed in **161.56 seconds** with all 9,999
logits rows and 29 spans matching the pinned streaming reference, maximum
absolute logit error `6.48e-5`, mean `4.79e-6`, and zero calibrated
outliers. The 11-second scalar path also passes after this change
(79.64 seconds, max `5.34e-5`, mean `6.44e-6`, zero outliers).
The tested AVX2/FMA softmax replaces `math.Exp` in vectorised attention;
it retains per-row F32 summation and uses the existing bounded-error
`ExpF32To` kernel. The 11-second and 100-second pinned logits and segments
still pass. Two more one-iteration 30-second samples measured
`20.50–20.57 s` versus `23.62–23.71 s` before the softmax change, with
unchanged approximate 7.742 GB request allocation. The complete
100-second SIMD request took **130.94 seconds**, returned 9,999 rows and
29 identical spans, and had maximum/mean logit error `6.87e-5`/`4.93e-6`
with zero values outside the original numerical gate. The 11-second
scalar path remains unchanged and passed in 73.04 seconds. Samples are
small, and the 100-second SIMD request is still slower than real time. The full 100-second
scalar run, labelled DER, masked intermediate encoder rows,
transfer-inclusive Vulkan/PTX latency, cancellation and production
readiness still need validation. The offline
PCM request retains its separate 376-row limit.

A four-worker checked prepacked GEMM now partitions complete 16-column
weight panels over disjoint output columns for bounded windows; it uses
the serial path on small jobs, without SIMD, or with overlapping raw
operands. Each output's accumulation order is unchanged. Two
three-iteration bounded-shape microbenchmarks found the four-worker path
faster than serial prepacking for 541×512×512, 541×2048×512 and
541×512×2048 projections. Race, malformed-input and numerical parity
tests pass. Two one-iteration 30-second PCM request samples fell from
`20.50–20.57 s` to `11.76–12.05 s`, while returned storage stayed around
7.75 GB (worker bookkeeping raised allocations from about 20,100 to
87,000 per request). The pinned **100-second PCM→logits→segments** run
completed in **77.90 seconds** at `GOMAXPROCS=4`, returning all 9,999
logits rows and 29 exact streaming spans. Maximum/mean logit errors stayed
`6.87e-5`/`4.93e-6`, with zero outliers under the existing gate. This is
one host and recording; concurrency throughput, labelled DER and full GPU
requests still require measurement.

A pinned real-podcast crop adds a different streaming audio input: the
300–320-second slice of tracked mono 16-kHz `testdata/podcast.wav`.
The independent low-latency PyTorch model and processor return 1,999
logit rows and one speaker span at `0–19.99 s`. The native five-second
PCM calls match all rows numerically (maximum/mean absolute error
`3.43e-5`/`4.83e-6`, zero values outside
`3e-4 + 2e-5*abs(reference)`) and emit the same span. The reference
itself marks this clip as single-speaker: matching it is model-oracle
parity, not labelled multi-speaker DER. The WAV digest verifies input
provenance only; it is not a generated-output acceptance gate.

To exercise a speaker-cache transition, a synthetic **31-second mixed
recording** concatenates the tracked 11-second JFK WAV with that podcast
crop, without resetting streaming state. The pinned low-latency processor
assigns speaker 0 to three JFK spans (`0.31–2.25`, `3.29–4.53`,
`5.40–10.61 s`) and speaker 1 to `10.98–30.99 s`. Native five-second
PCM calls match all four spans and all 3,099 pinned logit rows
(maximum/mean absolute error `2.67e-5`/`2.96e-6`, zero outliers under
the existing gate). This establishes parity through a model-assigned
speaker change, not labelled DER or natural multi-speaker conversation
quality.

The complete layer now reuses its internal attention-output buffer for
the residual: the public `Layer1Attention.ForwardOffline` still returns
two distinct owned outputs, while `Layer1Complete` only needs one owned
residual. A focused released-weight test verifies identical residual
values and unchanged input. Three matching-CPU 30-second streaming
request samples allocated `7.2356 GB/op` versus `7.7486 GB/op` on the
unchanged code, about **513 MB less per request**, and observed
`11.77–12.24 s` versus `13.63–16.27 s`. The timing ranges are noisy
and are not a sustained speed claim. A complete 100-second PCM→logits→
segments run still returns 9,999 rows and 29 pinned spans (79.97 s,
maximum/mean logit error `6.87e-5`/`4.93e-6`, zero outliers); the
podcast20 and synthetic mixed31 references also retain their pinned
spans and logit gates. No input or returned slice is reused across
requests.

`PCMStreamingRequest.AppendPCMContext` and `FinishContext` accept a Go
context without changing the existing methods. They check cancellation
before consuming PCM and between bounded 13-row encoder windows; a
cancelled request closes and returns no partial logits from that call.
Nil contexts and malformed PCM are rejected without advancing state. A
deterministic test cancels after the first window of a five-second call,
checks that its 72 emitted frames are withheld, and rejects any retry or
finish. Model kernels in progress cannot be interrupted; cancellation
granularity is one window. This is a functional cancellation check, not a
concurrent soak or service deadline guarantee.

An opt-in `-race` check loads the released model once, then shares its
immutable tower, head, frontend projection and compressor between two
independent two-second JFK streams. With 7,979- and 32,000-sample PCM
calls, both concurrent requests reproduce their serial 199-row logits;
previously returned logits stay unchanged. Separate mel, pending-window,
and speaker-cache state is retained for each request. This bounds
shared-model concurrency safety, not sustained throughput or cancellation
under concurrent load.
