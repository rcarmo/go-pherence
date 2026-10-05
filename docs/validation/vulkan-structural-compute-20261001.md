# Vulkan structural compute trials — 1 October 2026

Shared-input Q/K/V projection fusion, split-key online-softmax attention and joint K/V staging did not improve native latency on Sigma. All runtime prototypes were removed. The backend/model source remains the qualified `eb1582ce585febfb11ee6819d86d7ef6c848d6d5` tree; no trained-model or default changes are retained.

## Q/K/V projection fusion

Each encoder layer projects the same normalised input three times. The fused prototype stages one X tile and three separate weight tiles, accumulates Q/K/V together and writes separate existing output layouts. Both 32×32 and 64×64 output tiles preserve each output's increasing-K F32 arithmetic and bias addition. A third prototype maps three output bands into one dispatch while retaining the original single-projection register footprint; it does not share X loads across groups.

The baseline is the existing tile64 operator recorded as **three stages in one plan**, not three separate host submissions. The candidate is one stage in one plan. Each is timed with one dispatch-plan submission/fence. After warm-up, eight samples per arm use four alternating ABBA/BAAB blocks. Creation, uploads, downloads and correctness checks are excluded from timing.

| Full shape: M=1500, K=N=1280 | Three-stage baseline median (ms) | Fused median (ms) | Time increase |
|---|---:|---:|---:|
| Simultaneous QKV, tile32 | 25.853 | 28.632 | +10.75% |
| Simultaneous QKV, tile64 | 25.745 | 29.251 | +13.62% |
| Three output bands, tile64 | 25.752 | 27.651 | +7.37% |

All three paths matched every Q/K/V output bitwise on every sample. Independent F64 scalar checks, arena sentinels and cleanup passed for 1/1/1, 2/3/5, 31/33/65 and 65/63/31, plus the large shape. Tails include band boundaries within a workgroup. Simultaneous fusion doubles shared storage relative to the same tile's single projection and triples live accumulators; the measured trade-off is unfavourable. Register/shared-pressure attribution is a hypothesis, not a collected hardware-counter result.

## Split-key flash-style attention

The candidate partitions key tiles into 2, 4 or 8 ranges. Each range independently produces its online maximum, denominator and unnormalised weighted output; a second dispatch merges those summaries with maximum-adjusted exponential factors. The two dispatches are one ordered plan. Scratch is linear in queries, width and split count, with no quadratic score matrix. At 1500 queries, 20 heads and dimension64, scratch is 15.84, 31.68 or 63.36 MB respectively, excluding alignment.

This preserves F32 storage/accumulation but changes reduction grouping; numerical gates were retained rather than widened. Eight samples per arm follow warm-up in four alternating blocks. The baseline is the qualified score-ILP key32 operator.

| Splits | Baseline median (ms) | Split+merge median (ms) | Time increase |
|---|---:|---:|---:|
| 2 | 67.131 | 67.248 | +0.18% |
| 4 | 66.997 | 67.865 | +1.30% |
| 8 | 67.010 | 69.030 | +3.01% |

Full-shape maximum absolute difference was about `5.2–5.4e-8`, inside the unchanged `2e-5 + 2e-5*abs(reference)` bound. Independent F64 checks passed all eight shapes, including odd head dimensions, 4096-key and 4096-query extremes and empty split ranges. Sentinels and native cleanup passed. This prototype did not run trained transcription or the existing extreme-score stress corpus; no model-quality qualification is established.

First admission attempts rejected generated `UMin`, then boolean constants/logical-or instructions. Equivalent explicit branch and integer-state source avoided those instructions. The runtime admission envelope was not widened; failures are preserved. Final stored prototypes passed offline compilation and SPIR-V validation.

## Joint K/V staging

A third attention candidate stages K and V together for each key32 tile, eliminating the later V-loading phase and its barrier. Per-output arithmetic order remains the qualified score-ILP order. Shared memory increases from 14,528 to 22,720 bytes; no optional device feature is requested.

Six samples per arm in three alternating blocks gave `66.992 → 89.918 ms` (+34.22%). Full-shape output matched bitwise. The existing independent scalar shape/tail/softmax stress tests, sentinels and native cleanup passed. The extra shared footprint did not produce a useful measured trade-off and is rejected.

## Verification, resources and release

Starting revision: `eb1582ce585febfb11ee6819d86d7ef6c848d6d5`; Go `go1.26.2 linux/amd64`. These are synthetic native operator measurements, not acoustic tests, new cold-start comparisons or end-to-end speed acceptance. No model weights, decoder policy, serving profile or production process changed.

The user-authorised Intel runner retained four-CPU quota, `taskset -c 0-7`, 8 GiB memory/memory-plus-swap cap, no network and a read-only root. Qwen-idle and host available memory ≥6 GiB guards remained active. No cap, Vulkan feature or shader-contract relaxation occurred.

All prototypes and diagnostic test files were removed from runtime packages before final checks. `make model-layout-check host-build host-vet host-test docs-check`, affected Vulkan/Whisper race tests and source-diff verification passed on the unchanged retained code. No cross-build rerun was needed for the evidence-only tree. Final container exited 0 without OOM or guard abort. At 08:40 UTC, native/build/candidate work was drained, only the pre-existing `wrdp` containers remained, no Go process was active and Qwen slot0 was idle. `@llama` received explicit release. No independent review was completed for these rejected experiments.

## Evidence and remaining work

[Evidence directory](../../benchmarks/speech-foundations/vulkan-structural-compute-20261001/) retains GLSL/SPIR-V prototypes, external test sources as text, raw samples, admission failures, native/container states, checks, runner/options, provenance and hashes. `timings.json` gives medians, raw timing distributions and drift. Existing evidence paths must not be overwritten when reproducing the bounded diagnostics.

The inference deficit remains. Fusion and split parallelism alone are unsupported on these measurements. Next work needs direct original-graph/operator attribution with equivalent precision and timing boundaries, or another algorithmic change backed by measured cost. The original Vulkan+flash+VAD speed goal, independent acoustic/word timing and long-form/resume acceptance remain open.
