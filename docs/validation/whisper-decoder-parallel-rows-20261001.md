# Exact CPU decoder row scheduling — 1 October 2026

An explicit output-row split reduces fresh request medians by1.1–3.6% for three language fixtures and6.6–6.9% for VAD/word-timing fixtures. All tested logits, decoder caches, cross-attention observations and returned segment/word outputs stay bit-identical. The serial default and services are unchanged. The original-speed and independent quality targets are still unmet.

## Scheduling gap and retained opt-in

The [refreshed profile](whisper-workflow-refresh-20261001.md) identifies CPU dot/GEMM work as a substantial cost. `linearForwardOpt` already splits wide single-token projections, including the vocabulary head, across workers. `linearInto`, used by decoder self/cross Q/K/V/O and FFN projections, was serial despite using the same SIMD dot function.

`GO_PHERENCE_WHISPER_DECODER_PARALLEL_ROWS=1` enables the new `decoderLinearRowsInto` path for at least512 output cells. It partitions complete output rows among `min(linearWorkers, GOMAXPROCS, outDim)` workers. `linearWorkers` retains the existing `WHISPER_THREADS` selection. Other environment values, smaller projections and a one-worker budget use the existing serial loop. The separate int8 branch retains precedence.

Each output still uses the exact existing `simdrt.Sdot(x, weightRow)` and then the same optional bias addition. No dot reduction is split or reordered. SIMD kernels, activation/weight values, decoder logic, GPU encoder, cross-KV precompute and vocabulary-head implementation are unchanged. Input, immutable weight and bias slices must not overlap the decoder-owned output scratch. Extents are preflighted in the caller before spawning workers; malformed slices cannot cause a worker-side bounds panic. Every worker joins before the caller continues or reports cancellation.

The implementation creates bounded per-projection goroutines. No persistent worker pool, backend fallback, precision substitution or service resource change is introduced. The benchmark JSON records `DecoderParallelRows` and `DecoderProjectionWorkers`; flag-off reports1 worker, flag-on qualification reports4.

## Same-dot microbenchmark

Four-CPU guarded execution, deterministic inputs, 30 calls per sample, five alternating-order samples including the first timed sample. A reference call precedes timing for output comparison. Every output bit matches the serial SIMD reference.

| K/N | Two-worker change | Four-worker change |
|---|---:|---:|
|1280/1280 | −46.18% | −62.94% |
|1280/5120 | −43.97% | −60.01% |
|5120/1280 | −47.70% | −63.26% |
|1280/51866 | −40.62% | −57.71% |

The vocabulary-head row only establishes the existing output split's usefulness; that head was already parallel and receives no new gain from this change. Synthetic matrices have repeated values and hot-cache behaviour. Whole-request results provide a separate measured gate.

## Pinned decoder equality

The native test loads the verified original turbo Q5 model, uses deterministic1500×1280 encoded values and two independent decoder states. Eight fixed tokens execute serial and parallel paths alternately. Every one of51,866 logits per token, every self-K/V cache value, immutable cross-K/V values and960,000 captured cross-attention observations match by F32 bits. The final retained-tree rerun passes.

Pre-cancelled decoder-state construction returns cancellation; a subsequent fresh state produces exactly the serial first-token logits. This covers pre-cancellation/fresh reuse. In-flight cancellation, device faults and long-form recovery are not qualified by this test. The parallel helper joins its workers before returning; the existing checked decode cancellation boundary is unchanged.

Model-free tests compare exact SIMD results for six shapes, three bias lengths, worker counts0/1/2/4/7, tails and guards. Additional tests check exact `linearInto` opt-in dispatch, strict environment admission, a one-worker budget, concurrent independent owners and synchronous extent failures. No tolerance is widened. Existing model/decoder/word alignment regressions and whole-tree race tests pass.

## Fresh five-repeat request pairs

Same Intel GPU, CPU4/8GiB/no-swap, original pinned Q5 storage, outputILP attention/decode4 FFN, explicit locale and unchanged greedy checked decoder. Flag-off and flag-on processes use the same candidate executable. English/PT/FR pairs run baseline→candidate; VAD/word and groups pairs run candidate→baseline. Five requests include the first; no samples are removed and order is not randomised.

| Fixture | Serial baseline median (s) | Parallel median (s) | Change | Every returned output |
|---|---:|---:|---:|---|
| JFK English |6.320 |6.093 |−3.59% | Exact |
| MINDS Portuguese row0 |6.282 |6.088 |−3.09% | Exact |
| MINDS French row0 |5.760 |5.699 |−1.07% | Exact |
| JFK native VAD + words |7.516 |7.017 |−6.64% | Exact speech/audio spans, segments and word times |
| Two speech groups, VAD + words |15.192 |14.146 |−6.89% | Exact speech/audio spans, segments and word times |

The preliminary JFK run6.106s matches prior-window output but is retained only as pilot evidence. Fresh paired acceptance uses the separate6.093s candidate report with worker metadata. Model/input pins, language, VAD/word/gap flags, CPU/heap budgets, precision and encoder stats match within every pair. Only decoder scheduling differs.

Loading and preparation remain separate. JFK load2.878/2.877s and preparation0.895/0.894s are effectively unchanged. Five-request full arms:

| Fixture | Serial full arm (s) | Parallel full arm (s) |
|---|---:|---:|
| JFK |35.435 |34.282 |
| PT |35.287 |34.147 |
| FR |32.587 |32.232 |
| JFK VAD/words |41.446 |38.907 |
| Two groups VAD/words |79.637 |74.597 |

These gains are relative to the retained Go baseline. No fresh original-engine process runs in this window, and isolated percentages are not added to previous optimisations.

## Expanded output and allocation costs

Five flag-on requests each on extra MINDS PT rows1/2 (including43.76s), podcast20s noVAD and podcast20s nativeVAD/words match every retained outputILP baseline output. Those baselines come from an earlier window; they are output gates, not fresh speed pairs. Five silence/VAD/word requests return no transcript windows. Checked timestamp and word boundaries do not change.

The per-projection fan-out increases allocations. Fresh per-request median deltas:

| Fixture | Additional allocated bytes | Additional allocations |
|---|---:|---:|
| JFK |344,320 |9,910 |
| PT |331,512 |9,592 |
| FR |144,496 |4,176 |
| JFK VAD/words |770,456 |22,095 |
| Two groups VAD/words |1,555,912 |44,217 |

These are cumulative Go allocation counters, not RSS or retained-memory measurements. The extra allocation cost is included in measured request walls. Native encoder weights/scratch/stage/plan counts remain identical. No worker-lifetime/resource growth or persistent pool is introduced.

For JFK noVAD, decoder component medians change self0.176→0.110s, cross0.225→0.208s, MLP0.366→0.207s and head0.180→0.186s. The vocabulary head is unchanged. For JFK VAD/words they are0.394→0.246,0.502→0.468,0.818→0.460 and0.402→0.407s. Counters include teacher-forced alignment when enabled and are separate from full request speed acceptance. CPU four-worker matrix streams can still saturate memory bandwidth; only Sigma x86 native performance is measured.

## Verification and isolation

`make model-layout-check host-build host-vet host-test docs-check`, whole-tree race and marked ARM64/RISC-V builds pass. Native decoder tests execute on the final source; affected opt-in race and final documentation checks run after adding this report. No Vulkan shader or admission changes occur; the runtime inventory stays31 shaders.

The first candidate build encountered the package's two-argument `min` helper; nested calls fixed the demonstrated compile error. Host race invocation then reported `-race requires cgo`; race/build ran in the existing guarded C-toolchain container. The first trained invocation omitted HF metadata provenance and failed before model inference; the corrected environment and final rerun pass. Failed invocations do not enter timing results. Both read-only review delegates timed out after60s; no independent review approval is recorded.

The diagnostic-only microbenchmark test is removed from normal test discovery. [Hashed evidence](../../benchmarks/speech-foundations/whisper-decoder-parallel-rows-20261001/) retains its source, raw timings, trained tests, fresh/expanded JSON, baseline references, environment settings, parser/comparison, runner, binary hashes, state/exit/monitor records and full gates.

Fresh @llama coordination excludes competing experiments/builds/restarts. Every native test has a≤120s deadline. Qwen-idle and host-available-memory≥6GiB guards continue throughout builds and inference. All successful containers exit0 without OOM, are removed and actually drain before explicit release. Defaults, services, CPU/memory allocations, Qwen LAN and Gemma remain unchanged.

The [original workflow refresh](whisper-workflow-refresh-20261001.md) measured original JFK3.018s/noVAD and3.028s/VAD versus the slower Go path. This scheduling change does not establish original-speed parity. Its exact retained-Go outputs also preserve the existing VAD `Thank you.` disagreement; independent acoustic correctness, long-form/resume and fault gates still need qualification. Further decoder/device or original-compatible arithmetic work must preserve those gates before deployment.
