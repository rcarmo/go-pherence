# Whisper Turbo Vulkan Q8 projection placement — 13 September 2026

An aggregate packed-weight owner scales projection-only Q8 to the full 32-layer Whisper Turbo encoder. Pinned English, Portuguese and French transcripts remain exact while request time improves by 1.648–1.673× and reported resident encoder weight storage falls by 74.2%. Existing F32 constructors and serving defaults remain unchanged.

## Aggregate ownership

`VkLinearQ8WeightSet` owns one Q8 pipeline and one aligned host-visible/coherent allocation for up to 512 row-major matrices. Each matrix has independent packed-byte and F32-row-scale descriptor ranges. `VkF32Plan` receives only private, already-validated bindings and retains the shared owner buffer and pipeline through submission.

This replaces Tiny's original one-buffer/pipeline-per-projection implementation. Full Turbo has 192 projection matrices but still fits the unchanged native budget at 35 allocations. F32 projection tensors are omitted rather than duplicated.

Offline tests cover:

- matrix-count/shape/finiteness/cancellation admission before native allocation;
- exact nearest-even packing through the existing Q8 format;
- alignment and disjoint descriptor ranges for odd matrix shapes;
- shared pipeline/buffer use across views;
- per-view shape/index rejection;
- copied-owner invalidation;
- plan cancellation, retained aggregate owner, drain and reverse teardown.

## Trained short quality

Checkpoint: pinned `openai/whisper-large-v3-turbo` model SHA-256 `542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1`. The source checkpoint has 587 F16 tensors; inference activations, accumulation and output remain F32.

A full-width/depth four-frame probe compares:

1. original F32 projection weights;
2. F32 graph loaded with the exact Q8-dequantised weights;
3. actual aggregate packed-Q8 graph.

Results:

- packed-Q8 versus exact-dequantised graph maximum error: `1.9073486328125e-5`;
- packed-Q8 versus exact-dequantised RMS: `3.1618619061743235e-6`;
- Q8 versus original F32 maximum error: `0.09805488586425781`;
- Q8 versus original F32 RMS/reference-RMS: `0.04086124878048051`;
- three decoder prompt top-1 decisions remain `50360`.

Reported projection storage is `2,516,582,400` F32 bytes versus `630,620,160` Q8 bytes. Complete resident encoder weight storage is `2,540,369,920` versus `654,407,680` bytes. The Q8 resident graph reports `654,532,608` tracked Vulkan bytes in 35 allocations under the unchanged 4GiB/40-allocation budget.

## Pinned JFK quality and timing

Three full runs use the same 11-second JFK WAV and generation policy. F32 and Q8 encoders are constructed and run sequentially so both stay inside the same budget. Every Q8 run matches the corresponding F32 `WindowTranscript` exactly, including tokens and timestamps. Text remains:

> And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.

WER is 0/22 words.

| Trial | F32 request | Q8 request | Speedup |
|---|---:|---:|---:|
| 1 | 15.769s | 9.570s | 1.648× |
| 2 | 15.751s | 9.529s | 1.653× |
| 3 | 15.693s | 9.522s | 1.648× |

The final stricter resource run reports F32 `15.824s`, Q8 `9.494s`, and `1.667×`. It completed in `52.83s`, used `6,777,980KiB` maximum RSS and reported zero process swaps; host swap counters did not change.

Repeated native log SHA-256: `ac3fdf42cf728100df124ec439af90948571b2f65b841c2f68eb779353c7033e`. Final native log SHA-256: `a7316cb1ea0cd169999e763af5f616bfb437c0e090cecdf105a734add1c366fe`. Time output SHA-256: `cbbe2a5b57bc875462b27c330475cd06be5beb5222e51aa617008af8ed10889b`.

## Multilingual quality

The same aggregate owner was run sequentially against F32 on all three retained MINDS-14 source WAVs: Portuguese rows 0/1 and French row 0. Every Q8 `WindowTranscript` matches F32 exactly, including tokens and timestamps, and every transcript has 0 WER against its reference (20, 9 and 5 words). Combined three-clip request time is `46.014s` F32 versus `27.509s` Q8 (`1.673×`).

The multilingual run completed in `132.68s`, used `6,779,472KiB` maximum RSS and reported zero process swaps. Multilingual log SHA-256: `908fbe8e062ddde14e9a72bf957646d67242ac6cfe29fda0e21d0e3261b95e35`. Time SHA-256: `b74d72f1067e36d6ea48d61d3dc26ae0ef434918fa8b7a186701e270b4165827`.

## Robustness and exact-timestamp placement

A separate 63-second composition places JFK speech at 2s and 42s, producing three 30-second windows with an exact-zero final tail. It also tests five seconds of digital silence with default policy and with `SkipDigitalSilence`.

Full projection Q8 preserves all text/tokens and 0 WER, but changes one first-window timestamp boundary from `11.0s` to `11.2s`; it therefore fails the strict exact-output gate. Default silence remains exactly equal to F32 but both paths hallucinate “Thank you.”. The opt-in exact-zero skip remains empty and exact.

Projection-family dequantised screens localize the timestamp shift to Q and O independently. K and V remain exact, as does MLP-only Q8. The broad selective packed candidate therefore quantises K/V/FC1/FC2 and retains Q/O in F32. It preserves every token and timestamp on all three robustness fixtures.

| Path | Reported encoder weight bytes | Three-fixture time | Speedup | Exact timestamps |
|---|---:|---:|---:|---|
| F32 | 2,548,039,680 | 46.456s | 1.000× | yes |
| Full Q8 | 662,077,440 | 27.924s | 1.664× | no |
| MLP-only Q8 | 1,290,567,680 | 33.791s | 1.375× | yes |
| K+V+MLP Q8 | 976,322,560 | 30.790s | 1.509× | yes |

The final four-path robustness process used `5,805,996KiB` maximum RSS and zero process swaps. Full-Q8 timestamp shift is retained as a hold, not hidden by a looser comparison.

Final robustness log SHA-256: `960ec87bf871eaf3d02ad003af07a58c488511241138fa8a80ab0c722266396d`. Time SHA-256: `3ea33d6081af337e88393ba774a59ca10319bc41b2db8ec97900564223a1b772`. The earlier full/MLP comparison remains retained for provenance.

## Decision

Full projection Q8 passes pinned English/PT/FR text/token gates and is the fastest candidate, but remains held because of the multi-window timestamp change. Explicit K+V+MLP Q8 is the broadest retained exact-output candidate at `1.509×` and 61.7% lower reported encoder-weight storage. MLP-only Q8 remains a narrower exact comparison. None changes serving/default selection. Natural noisy/long-form inputs, broader WER, energy/stress, recovery and Community-1 placement remain open.

## Reproduce

Short trained gate:

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
export GO_PHERENCE_WHISPER_TURBO_DIR=/path/to/pinned/whisper-turbo
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-turbo-q8-weight-check
```

Add the full pinned JFK and MINDS phases:

```sh
export GO_PHERENCE_WHISPER_JFK_PATH=/path/to/pinned/jfk.wav
export GO_PHERENCE_MINDS_FIXTURE_DIR=/path/to/pinned/MINDS-cache
export GO_PHERENCE_TEST_VULKAN_TURBO_Q8_SPEECH=1
export GO_PHERENCE_TEST_VULKAN_TURBO_Q8_MULTILINGUAL=1
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-turbo-q8-weight-check
```
