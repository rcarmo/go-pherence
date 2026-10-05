# Vulkan projection follow-up — 1 October 2026

The qualified score-ILP encoder still spends most separately fenced stage time on projections. None of this follow-up's five candidates improved latency enough to retain. All temporary backend changes were removed; production and encoder defaults are unchanged.

## Refreshed attribution

Starting revision: `445b6cadd566a8b11d021020dccac2c916b20773`. The stage profiler now explicitly admits the default F32 backend and the qualified `vulkan-f32-tile64-key32-scoreilp` mode; unknown modes fail before loading models. A model-free admission test covers these selections. The profiler creates individually fenced plans from the same encoder stages and verifies that their final output exactly matches the whole graph.

Three passes on pinned JFK and Turbo F16 checkpoint widened to F32 gave these mean per-pass totals:

| Operator | Seconds |
|---|---:|
| F32 projections | 3.600 |
| F32 key32 score-ILP attention | 2.160 |
| Convolutions | 0.120 |
| GELU | 0.075 |
| Add | 0.077 |
| Normalisation | 0.067 |

This attribution changes fencing/submission costs and excludes CPU decoding. Its sum is not a whole-request time. Prior [matched request measurements](vulkan-attention-scoreilp-20261001.md) remain the acceptance evidence. Fresh attribution supports investigating projections rather than adding more attention tile variants.

## Rejected candidates

Projection candidates expanded the 16 output accumulators and four-by-four row/column loops into static scalar registers. K32 order and tile geometry stayed unchanged. One variant scheduled rows first, the other columns first. This tests register/index overhead and independent arithmetic scheduling without K unrolling or reduced precision.

| Candidate | 1500/1280/1280 | 1500/1280/5120 | 1500/5120/1280 |
|---|---:|---:|---:|
| Static scalar registers, rows first | +3.99% | +5.04% | +2.97% |
| Static scalar registers, columns first | +3.96% | +6.06% | +3.26% |

Positive values are increased host dispatch/fence time. Each shape used eight samples per arm in four alternating ABBA/BAAB blocks after warm-up, within one process. Setup, transfers and checks were outside timing. Full-shape results matched the existing tile64 kernel bitwise on every measured run. Twelve small/tail geometries passed the existing scalar-reference tolerance, sentinel and cleanup checks. Both candidates are rejected.

Attention candidates retained increasing-channel score FMA and increasing-key value order:

| Candidate | Prior score-ILP median (ms) | Candidate median (ms) | Time change |
|---|---:|---:|---:|
| Four scores/lane, eight X lanes | 66.970 | 74.007 | +10.51% |
| Two-component score vector | 66.870 | 66.990 | +0.18% |
| Four-component output vector | 66.960 | 66.886 | −0.11% |

These used six samples per arm in three alternating blocks at 1500×20×64. Full output matched bitwise; existing independent scalar/stress/tail tests, sentinels and cleanup passed. The four-score variant trades lanes for more per-lane registers and did not win. Vector forms are effectively tied at this measurement scale and are not retained. No trained-model arms were run for rejected kernels.

The first score-vector source failed admission for `OpCompositeConstruct` (opcode 80), before dispatch. Fixed component assignments generated already admitted instructions and passed. The runtime opcode envelope was not widened. This failure and both sources are retained. The original engine's non-cooperative matrix shader uses paired vector caches/dot products; these limited vector probes do not qualify that packed arithmetic or its rounding semantics.

## Verification and isolation

Go was `go1.26.2 linux/amd64`; model/input pins and exact runner settings are in the archived stage environment. The authorised Intel container retained a four-CPU quota, `taskset -c 0-7`, 8 GiB memory/memory-plus-swap cap, `GOMEMLIMIT=4GiB` for trained attribution, no network and a read-only root. Qwen-idle and host-available-memory ≥6 GiB guards remained active. No service, default, device feature, precision or resource limit changed.

After removing prototypes, the retained profiler-mode change passed ten repeated admission tests, affected vet, `make model-layout-check host-build host-vet host-test docs-check`, Whisper race tests and ARM64/RISC-V package builds. Foreign builds provide compilation evidence only. The isolated check exited 0 with `PASS_FINAL_CHECKS`, no OOM and no guard abort. Static prototype modules passed `spirv-val`; no embedded shader was changed.

A delegated source-comparison attempt timed out at 100 seconds. No independent approval is established for this follow-up. At 05:52 UTC, native/build/container work was drained and `@llama` received explicit release. Only the pre-existing `wrdp` containers remained.

## Evidence and next work

[Evidence directory](../../benchmarks/speech-foundations/vulkan-projection-followup-20261001/) retains stage-level samples/totals, candidate GLSL/SPIR-V, external-shader diagnostic source as text, logs, failed admission, runner, environment and container states. `timings.json` contains the medians and changes above. Existing evidence paths must not be overwritten when reproducing these guarded diagnostics.

The next speed hypothesis needs packed/vector projection arithmetic with an explicit precision contract, independent decoding/reduction tests and trained whole-request measurements. Repeating these scalar-register or shared-layout probes is unsupported by the measurements. The original-engine speed goal and independent acoustic/word-timing/long-form/resume acceptance remain open.
