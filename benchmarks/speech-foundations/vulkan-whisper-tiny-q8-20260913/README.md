# Whisper Tiny Vulkan Q8 projection placement — 13 September 2026

An explicit resident Whisper encoder now supports per-output-row symmetric Q8 for transformer projection matrices. On the pinned Tiny checkpoint, it reduces reported encoder weight storage by 64.5%, preserves the pinned JFK transcript exactly and improves whole-request time by 12.7–14.4% across three repeated paired trials. Existing constructors, model graphs and serving defaults remain F32.

## Placement and ownership

`NewVulkanEncoderQ8Weight` replaces only each encoder layer's Q/K/V/O/FC1/FC2 projection storage and dispatch. Stem convolutions, positions, biases, LayerNorm, GELU, attention arithmetic, residuals, accumulation, outputs, cross-K/V generation and decoder remain F32.

All projections share one immutable aligned packed-weight allocation and one Q8 pipeline. Each matrix has separate packed-byte and F32-scale descriptor ranges. A private validated plan-stage binding carries those ranges into `VkF32Plan`; callers cannot forge private bindings across packages. Plan preflight retains both arena and packed buffers through submission. Cancellation, drain, copied owners and reverse teardown are covered. F32 projection tensors are omitted from Q8 weight arenas rather than duplicated.

The existing `NewVulkanEncoder` path is unchanged. No environment toggle, implicit fallback or server/default selection is added.

## Trained quality

Checkpoint: pinned `openai/whisper-tiny` revision `169d4a4341b33bc18d8881c4b69c2e104e1cc0af`, model SHA-256 `7ebd0e69e78190ffe1438491fa05cc1f5c1aa3a4c4db3bc1723adbb551ea2395`.

A deterministic 34-frame trained probe compares three paths:

1. original F32 projection weights;
2. F32 graph loaded with the exact Q8-dequantised weights;
3. actual packed Q8 graph.

The actual packed graph differs from the exact dequantised-weight F32 graph by at most `2.193450927734375e-5` at the final encoder output. This is kernel rounding, separate from quantisation loss. Against the original F32 graph, final encoder maximum error is `0.3406410217285156`; RMS error/reference-RMS is `0.04225807720811263`. Three decoder prompt steps retain the same top-1 tokens: `50362`, `50358`, `11`.

The pinned 11-second JFK fixture (SHA-256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`) produces exactly the baseline `WindowTranscript`, including tokens and timestamps, in every warmup and timed run. Text remains:

> And so my fellow Americans ask not what your country can do for you, ask what you can do for your country.

Fixture WER is 0/22 words.

This is one English fixture plus a short synthetic trained probe. It does not qualify Turbo, multilingual speech, broad WER, silence/noise, long recordings or Community-1.

## Storage and timing

Reported encoder weight storage includes stem/normalisation/position/bias tensors and all projection allocations:

| Path | Weight bytes |
|---|---:|
| F32 resident encoder | 32,839,680 |
| Q8 projection resident encoder | 11,661,312 |

Projection matrices alone are `28,311,552` F32 bytes versus `7,133,184` Q8 bytes including row scales/alignment.

Each of three trials used an initial validated call then four whole-request samples per path in two alternating blocks. Timing includes frontend, resident encoder upload/run/download, cross-K/V and decoder; media/model load is outside the samples.

| Trial | F32 median | Q8 median | Speedup |
|---|---:|---:|---:|
| 1 | 548.710ms | 487.025ms | 1.127× |
| 2 | 548.865ms | 479.764ms | 1.144× |
| 3 | 553.065ms | 487.355ms | 1.135× |

The final resource-measured run reports F32 `550.806ms`, Q8 `484.046ms` and `1.138×`; it took `6.18s`, used `397,340KiB` maximum RSS and reported zero process swaps. Host swap counters were unchanged during that run.

Repeated native log SHA-256: `e89ebb9618f23078f65dc9827230dca3c6b44946a7baab818a21e8404c3b3030`. Final native log SHA-256: `dcc055d9294016d3c62ab688a2d98f27cb3a949e7af1d14baa42e1a22069351b`. Time output SHA-256: `b703a230b60bcc01926baccba5eb8d40003333e92f2eebeb45c69d40a708a622`.

## Decision

Tiny projection-only Q8 passes this first trained quality/performance checkpoint and remains an explicit candidate. It is not a production/default promotion: the full plan still requires broader multilingual quality, long/noisy/silent inputs and resource scaling. The later Turbo checkpoint replaces the original per-projection ownership with one aggregate allocation/pipeline; this report's numerical and timing results remain valid.

## Reproduce

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
export GO_PHERENCE_WHISPER_TINY_DIR=/path/to/pinned/whisper-tiny
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-whisper-q8-weight-check
```
