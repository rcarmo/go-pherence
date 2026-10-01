# Exact CPU four-dot trial — 1 October 2026

The existing AVX2 four-dot helper improves a synthetic FFN-down kernel by18%, but fresh whole-request changes range from0.36% faster to0.15% slower on three language fixtures and0.7–1.0% faster with VAD/words. The second decoder opt-in is rejected for insufficient demonstrated benefit on these fixtures. The accepted four-worker scheduling option remains intact.

## Existing kernel reuse

The accepted [parallel-row decoder](whisper-decoder-parallel-rows-20261001.md) runs one `Sdot` per output row. This trial groups four neighbouring output rows, using `Sdotx4(activation, weightRows, inDim)`. The shared activation vector is loaded once for four weight streams. Each output still consumes its own two AVX2 accumulator chains; the final pairwise vector/horizontal reductions match `Sdot` for K divisible by16.

The separate `DotRowsx4` helper accumulates with a different reduction schedule and is excluded. No new assembly, SIMD reduction rewrite, weights, activation quantisation or precision mode is introduced. The trial requires amd64, the existing ISA gate and K divisible by16; other architectures/tails retain ordinary `Sdot`. The vocabulary head already uses a separate parallel path and is unchanged.

Temporary admission uses `GO_PHERENCE_WHISPER_DECODER_ROWS_X4=1` together with the accepted `GO_PHERENCE_WHISPER_DECODER_PARALLEL_ROWS=1`. Output cells are grouped inside each existing worker's bounds; remaining cells use the baseline dot. Bias addition remains per-cell and all workers join. Normal defaults and services stay unchanged. This second flag is removed after rejection and has no retained runtime effect.

## Operator diagnostic

Deterministic sinusoidal finite F32 inputs/weights, existing four-worker row partition, 30 calls per sample, five alternating-order samples including the first. Every output bit matches four independent `Sdot` calls, with short projection/output-tail cases included.

| K/N | Independent-dot median (ms/30 calls) | Four-dot median (ms/30 calls) | Change |
|---|---:|---:|---:|
|16/7 |0.062 |0.066 |+5.96% |
|32/519 |0.167 |0.106 |−36.73% |
|1280/1280 |2.421 |2.286 |−5.57% |
|1280/5120 |14.312 |14.099 |−1.49% |
|5120/1280 |15.703 |12.860 |−18.10% |
|1280/51866 |216.626 |215.179 |−0.67% |

The head-shaped row is diagnostic only; the vocabulary-head implementation never uses the candidate. The FC2 result is a real improvement in this synthetic operator workload. It does not establish a corresponding whole-request gain.

Model-free tests compare bit patterns across six shapes, three bias lengths and worker counts1/2/4/7, including guards and odd-K baseline paths. Another test checks the same two-chain reduction against `Sdot` for K16/32/64/1280/5120 across five finite input patterns, with signed zero/subnormal values included. Existing ISA gates and strict environment admission are checked; no numerical tolerance is introduced. Race/vet pass on these diagnostic tests.

## Pinned decoder test

The native test uses the verified original turbo Q5 model and deterministic1500×1280 encoded input. Two independent states run eight fixed tokens with the four-worker baseline and four-dot trial. Every51,866-logit vector, self-K/V cache, immutable cross-K/V value and960,000 cross-attention observation matches by F32 bits. Pre-cancelled state construction and fresh first-token reuse pass.

These tests cover the checked finite/ISA workload and short trained token sequence. They do not qualify nonfinite-input payload behaviour, full encoder hidden states, in-flight faults, long-form/resume or independent acoustic correctness. Expanded PT/podcast recordings were not run because the request gain did not justify retaining another dispatch option.

## Fresh five-repeat requests

Same candidate executable with the second flag off/on, accepted four-worker decoder enabled in both, retained outputILP/decode4 Vulkan encoder, original pinned weights, four CPU threads and heap4GiB. Original-engine inference is not run in this window. Language arms use baseline→candidate processes; VAD/word and groups arms use candidate→baseline. Five requests include the first; no samples are discarded or order randomised.

| Fixture | Four-worker baseline (s) | Four-dot candidate (s) | Change | Outputs |
|---|---:|---:|---:|---|
| JFK English |6.088 |6.067 |−0.36% | Exact |
| Portuguese row0 |6.084 |6.078 |−0.10% | Exact |
| French row0 |5.655 |5.663 |+0.15% | Exact |
| JFK native VAD + words |7.016 |6.966 |−0.71% | Exact spans/segments/word times |
| Two groups native VAD + words |14.170 |14.031 |−0.98% | Exact spans/segments/word times |

Every returned output across all five requests matches the baseline; model/input pins, precision, encoder stats, CPU/thread/heap and VAD/word/gap settings are equal. Metadata confirms four workers and the candidate flag selection. Decoder MLP counters decrease, but encoder, other decoder work and request orchestration limit the end-to-end change. Confidence intervals and hardware clock controls are not collected, so a universal speedup or regression is not established.

The rejection is a maintenance choice: this tested workflow does not demonstrate enough benefit to retain a second opt-in beside the accepted worker split. The synthetic FC2 benefit could matter in another workload; these measurements do not rule that out. The original≈3s versus Go≈6s single-window target and VAD `Thank you.` disagreement remain unresolved.

## Retained source and checks

The temporary flag/dispatch, benchmark metadata and all x4 diagnostic tests are removed. `decoder_rows.go` and `performance_goal_test.go` restore the exact accepted `a0aae902` source. The existing SIMD helpers are unchanged. Candidate source/patch/tests remain under [hashed evidence](../../benchmarks/speech-foundations/whisper-decoder-x4-20261001/) with raw micro/native/request JSON, settings, parser, provenance, runner, states/exits and verification logs.

A read-only methodology judge completed on the supplied measurement facts. It confirms the scope and notes that rejecting another opt-in is a maintenance decision. It did not inspect source/raw evidence or provide acoustic acceptance.

The retained tree passes `make model-layout-check host-build host-vet host-test docs-check`, affected Vulkan/Whisper race tests with the accepted scheduling option enabled, and marked ARM64/RISC-V builds. Final documentation checks follow this report. No new shader enters the31-shader inventory.

All successful diagnostic/request/gate containers exit0 without OOM and actually drain before release. The fresh @llama isolation window uses CPU4/8GiB/no-swap, Qwen-idle and host-available-memory≥6GiB guards; native tests have≤120s deadlines. Qwen LAN, Gemma, defaults/services and resource allocations remain unchanged. No deployment or goal completion follows this trial.
