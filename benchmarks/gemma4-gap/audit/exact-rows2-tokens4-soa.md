# Exact two-row/four-token SoA audit

## Decision

Reject the exact two-row/four-token production candidate and retain `ea90f25b264939b0da3b4065733521e882f27181`.

The candidate has a real local mechanism: for the same sixteen outputs it halves Q8 loads and unsigned-Q4 correction dots while preserving the legacy eight-lane FP32 accumulation. It won 8 of 10 clean, order-balanced, packing-inclusive projection pairs, with 1.11--1.14% lower raw medians. That was sufficient to run the model gate, not sufficient to bypass it.

All exact state and token-ID gates passed, but the five-pair frozen request rejected promotion. The candidate won only two prompt pairs, lost three, and had a 1.36% median paired regression. Its 0.66% higher unpaired median was smaller than the observed order and frequency variation and does not override adjacent-pair direction. No production source from the candidate is retained; the rejected implementation is preserved as [`exact-rows2-tokens4-soa.patch.gz`](exact-rows2-tokens4-soa.patch.gz).

## Post-promotion target

A phase-only profile of the promoted `ea90f25b` checkpoint sampled 9.52 CPU-seconds during a 2.352-second prefill. `dotQ4_0Q8_0Tokens8SoAVNNI` accounts for 7.87 CPU-seconds, or **82.67%** flat. The four-token tail accounts for 5.04%, while the new AVX2 Q8 block quantiser is only 1.47%. Direct Q8 preparation has therefore moved packing out of the leading costs and made the exact dot topology the unambiguous target.

The profiled run reached 52.730578 prompt tok/s under clean load, but it is attribution evidence rather than a replacement for the accepted paired median of 49.152 tok/s. Applying the 82.67% sample share to that accepted median gives an estimated 2.0856 seconds in the hotspot and 0.4372 seconds elsewhere. Reaching the 89.405 tok/s gate with the remainder fixed requires the hotspot to fall to 0.9497 seconds -- approximately **2.196 times faster**. Matching the 91.2296 tok/s oracle requires approximately 2.262 times hotspot speed.

Raw evidence is [`go_prefill_ea90_exact_batch_profile.log`](go_prefill_ea90_exact_batch_profile.log), [`go_prefill_ea90_exact_batch_top.log`](go_prefill_ea90_exact_batch_top.log), [`go_prefill_ea90_exact_batch_load.log`](go_prefill_ea90_exact_batch_load.log), and [`go_prefill_ea90_exact_batch.pprof`](go_prefill_ea90_exact_batch.pprof).

## Equal-work mechanism

The retained kernel evaluates one Q4 row against eight Q8 tokens. The candidate evaluates two Q4 rows against four tokens, then repeats for tokens four through seven. Both therefore make two assembly calls and produce sixteen outputs. Source instruction lines count each five-byte `VPDPBUSD` byte sequence as one instruction.

| Per QK block for sixteen outputs | Two retained 1-row/8-token calls | Two candidate 2-row/4-token calls |
|---|---:|---:|
| Assembly calls | 2 | 2 |
| Loop source instruction lines | 172 | 170 |
| Q4 scale and payload bytes requested | 36 | 72 |
| Q8 scale and payload bytes requested | 576 | 288 |
| Total source bytes requested | 612 | 360 |
| Q4 `VPDPBUSD` dots | 16 | 16 |
| Correction `VPDPBUSD` dots | 16 | 8 |
| Total `VPDPBUSD` instructions | 32 | 24 |
| FP32 block FMAs | 16 | 16 |
| Live output accumulators per call | 8 | 8 |
| Local scale spill per call | 0 bytes | 32 bytes |

The candidate reduces requested block input by 41.2% and VNNI dot instructions by 25%. It pays by loading and decoding both Q4 rows twice, once for each token half, and by spilling eight combined Q4/Q8 scales because all sixteen YMM registers are occupied. The assembled function is 868 bytes versus 806 bytes for the retained function despite its one-line-shorter loop.

The existing direct SoA Q8 layout is used unchanged. There is no extra packing traversal, transient representation, allocation, or model-weight copy. Odd output rows fall back to the retained eight-token kernel, and four-token and scalar token tails remain on their retained paths.

## Exactness gate

The assembly optionally exposes its eight by eight FP32 lane state after the final block and before reduction; production passes a nil probe pointer. A portable reference exercised both token halves, five deterministic random samples, and every block count from 1 through 80. Every one of the 64 lane states and all eight reduced outputs matched bit-for-bit in all 800 cases.

A projection oracle compared the production traversal directly with the retained one-row/eight-token kernel at one, eight, and eighty blocks; one, two, 65, and 513 output rows; and batches 64, 65, and 124. This covers odd-row fallback, four-token/scalar token tails, 64-row task tails, and dynamic chunk claims. Every output bit matched. Malformed row stride, raw, tile, block, token-base, and output shapes were rejected.

The following gates passed:

* `go test ./loader/gguf -count=1`;
* focused lane, projection, and dequant-oracle tests, 20 repetitions;
* the focused race suite, three repetitions;
* `go vet ./loader/gguf`;
* arm64 and riscv64 test compilation;
* `git diff --check`.

Commands and output are in [`go_q4_rows2_tokens4_soa_correctness.log`](go_q4_rows2_tokens4_soa_correctness.log).

## Packing-inclusive projection

Both series ran pinned to CPUs 0--5 with `GOMAXPROCS=6`, alternating retained/candidate order. `BenchmarkQuantMatrixProjectBatchQ4_0Gemma4Shapes/out10240/batch124/batched` includes direct F32-to-SoA Q8 preparation and the complete projection. Process sampling reported no non-benchmark hot process; allocation remained 19 allocs/op and approximately 365 KB/op.

| Series | Retained median | Candidate median | Raw median delta | Candidate pair wins | Median paired delta |
|---|---:|---:|---:|---:|---:|
| Five 1-second pairs | 9.901387 ms | 9.788742 ms | -1.14% | 4/5 | -1.02% |
| Five 3-second pairs | 9.800424 ms | 9.691443 ms | -1.11% | 4/5 | -1.46% |

The last two long pairs slowed for both binaries under sustained load, but order was balanced and the candidate retained a four-of-five direction. This is a small, repeatable projection improvement with unchanged packing cost, so it crossed the threshold for the expensive model gate. Raw output and process samples are [`go_q4_rows2_tokens4_soa_projection_paired.log`](go_q4_rows2_tokens4_soa_projection_paired.log) and [`go_q4_rows2_tokens4_soa_projection_load.log`](go_q4_rows2_tokens4_soa_projection_load.log).

## Frozen 124+48 request

Three standalone candidate runs first reproduced all 48 frozen IDs. Five adjacent, order-balanced retained/candidate pairs then repeated the exact request under pinned six-CPU execution; every run reproduced the same IDs and K/V trajectory.

| Pair | Retained prompt tok/s | Candidate prompt tok/s | Candidate delta |
|---:|---:|---:|---:|
| 1 | 52.432923 | 52.781229 | +0.66% |
| 2 | 52.265844 | 51.556736 | -1.36% |
| 3 | 54.910430 | 53.872427 | -1.89% |
| 4 | 52.166529 | 52.924391 | +1.45% |
| 5 | 54.300135 | 51.910482 | -4.40% |
| Median | **52.432923** | **52.781229** | **+0.66% unpaired** |

The paired median is **-1.36%**, and the candidate wins only two of five pairs. Decode uses batch one and never enters the candidate path; its retained and candidate medians of 9.521717 and 9.631373 eval tok/s therefore quantify noise rather than an effect of this change.

Even the candidate's unpaired prompt median remains 40.96% below the 89.405 gate and would need another 1.694 times whole-prompt speed. The small projection floor does not survive the complete graph consistently enough to retain. Raw output and process samples are [`go_prefill_rows2_tokens4_soa_paired_r5.log`](go_prefill_rows2_tokens4_soa_paired_r5.log) and [`go_prefill_rows2_tokens4_soa_load.log`](go_prefill_rows2_tokens4_soa_load.log).

## Consequence

The experiment confirms that sharing four Q8 tokens across two rows is valid and locally useful, but the AVX2 register limit forces a second Q4 pass and a scale spill. A roughly 1% projection gain cannot close a hotspot that now needs approximately 2.196 times speed and is not stable across the complete request. Future exact work should not repeat this row/token split unless it removes the duplicated Q4 decode or materially changes the register and call topology.
