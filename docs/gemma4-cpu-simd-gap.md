# Gemma4 E4B CPU SIMD gap

This note tracks the exact CPU optimisation work for the Gemma4 E4B Google QAT Q4_0 model at `models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf`. The canonical benchmark contract, raw oracle evidence and current reproduction commands live in the [Gemma4 CPU performance-gap programme](../benchmarks/gemma4-gap/README.md).

All accepted measurements use six pinned Intel Core i7-12700 logical CPUs (`taskset -c 0-5`, `GOMAXPROCS=6`). Optimisations must preserve bit-exact activations, logits, generated tokens, K/V state, blockwise FP32 accumulation, eight lane accumulators and the legacy final reduction order.

## Accepted comparison contract

The frozen request evaluates BOS token `2` plus 123 copies of token `10979`, for 124 timed prompt tokens. Prefill returns the first output as an untimed generation boundary; generation timing then covers exactly 47 subsequent evaluations. The gate checks all 48 Go output IDs, including the boundary token.

The accepted oracle is a CUDA-disabled rcarmo/llama.cpp b607 (`065d9d501`) build with CPU flash attention, F32 K/V and the exact Go evaluated-token trajectory replayed after the prompt. Six adjacent samples establish these medians and 98% gates:

| Phase | CPU-only llama.cpp oracle | 98% gate |
|---|---:|---:|
| 124-token prompt | 91.2296 tok/s | 89.4050 tok/s |
| 47 generation evaluations | 10.5265 eval tok/s | 10.3159 eval tok/s |

The former 447.166 prompt tok/s and 9.310 decode tok/s comparison is invalid. It came from a CUDA-enabled binary where `-ngl 0` did not produce CPU-only execution. `-ngl 0` is used in the accepted command only after the audit binary proves that it has no GPU backend (`"backends": "CPU"`, empty `gpu_info`).

## What is retained

Long prefill uses an AVX-VNNI one-weight-row/eight-token kernel. Each output keeps the legacy eight FP32 lane accumulators, blockwise FMA and final reduction order, so the batched path remains bit-identical to sequential execution. Each eight-token Q8_0 input tile uses a structure-of-arrays layout: eight contiguous FP32 scales followed by eight contiguous Q8 vectors. One packed scale load and multiply replaces eight scalar scale sequences without changing arithmetic. Randomised 1--80-block comparisons are bit-exact.

Q8_0 activation quantisation writes directly into caller-owned storage across contiguous worker spans. The parallel and serial paths produce identical scales and quantised bytes, retain oversized-slice behaviour and pass the race detector. This work previously established a 38.808 prompt tok/s median.

The retained Q4 prefill kernel now dots unsigned nibbles and computes `8*sum(Q8)` dynamically with a second `VPDPBUSD`, then subtracts the int32 correction before the existing FP32 conversion, scale FMA and reduction. It is algebraically identical to `(q4-8)*q8` but removes the Q4 `VPSUBB` and all eight per-token `VPABSB`/`VPSIGNB` transforms. Tightly paired trials reduced kernel time by approximately 16.7%. Three alternating complete-request pairs moved median prompt throughput from 40.160 to 48.875 tok/s (+21.7%), with all 48 IDs unchanged.

Decode retains the fused Q6_K multi-block AVX-VNNI GEMV and the eight-row Q4_0 kernel with a precomputed per-block `8*q8` correction. The new B64+ prefill kernel does not affect that route; paired decode medians were effectively unchanged at 9.128 baseline versus 9.161 candidate eval tok/s.

| Runtime | Prompt | Generation | Oracle efficiency |
|---|---:|---:|---:|
| CPU-only llama.cpp b607, exact phases | 91.230 tok/s | 10.526 eval tok/s | 100% / 100% |
| Go, retained median | 48.875 tok/s | 9.161 eval tok/s | 53.6% / 87.0% |

Neither phase currently passes its corrected 98% gate. Decode is 13.0% below the oracle; prefill remains 46.4% below it.

## What was rejected

The four-weight-row/two-token Q4_0 tile preserves exact reduction order but reduced prefill to 20.53--21.15 tok/s. Rewriting it around the unsigned-Q4 correction narrowed the isolated-kernel loss, but it still lost in the real projection path. A two-row/four-token tile improved its isolated kernel by 2.6--4.2%, then lost four of five end-to-end pairs by 4.9--7.6% because extra weight passes and scatter overhead outweighed activation reuse.

Splitting the retained eight-token tile into two four-token passes freed four YMM registers and allowed two VNNI dependency chains to overlap. It remained exact, but longer pinned samples regressed from an 812 ns median to 1,009 ns because every Q4 block had to be loaded and decoded twice.

Gathering eight activation scales into a packed vector was exact but 1--4% slower in four of five paired runs. The retained structure-of-arrays layout obtains the same packed scale arithmetic from one contiguous load. Flat-interleaved activation and persistent-worker variants also regressed the complete request.

A direct signed-byte `VPDPBSSD` replacement is unavailable on this Alder Lake host. `avx_vnni` provides `VPDPBUSD`; signed-byte `VPDPBSSD` requires AVX-VNNI-INT8. Dynamic unsigned-Q4 correction is the retained way to use the available dot instruction without `VPABSB`/`VPSIGNB`.

Precomputing the correction as eight `int16` lanes per token was 4.2--6.3% faster than dynamic correction in the isolated kernel. It enlarged an 80-block activation tile from 23,040 to 33,280 bytes and added packing work, however, and lost every full-request pair. Compact and dynamic medians were 40.392 and 44.863 prompt tok/s respectively, so the compact form was reverted.

A llama.cpp-style packed reduction is not an exact substitute for the legacy contract. It reduces a complete 32-element block to one integer before FP32 accumulation, while go-pherence keeps eight FP32 partial accumulators until final reduction. A deterministic random probe differed in 877 of 1,000 cases. Its eight-weight-row/one-token orientation was also approximately 1.35 times slower than the retained one-weight-row/eight-token orientation under the exact reduction constraint. Fully unrolling the fused Q6_K block dot was exact across 1,000 random blocks, but increased its median from 27.79 ns to 29.49 ns.

## Where the time goes

The pre-correction profile that selected this target attributed 83.48% of flat CPU samples to `dotQ4_0Q8_0Tokens8SoAVNNI`. A new phase-only profile of the retained unsigned-correction binary attributes 10.87 of 13.28 sampled CPU seconds, or 81.85%, to the same function. `quantizeQ8_0To` is only 4.07% flat and the four-token tail kernel is 4.74%. The profiled request ran slowly under shared host load, so its 19.156 prompt tok/s is not a throughput sample; the profile is evidence about attribution only. The raw artefacts are [`go_prefill_retained_unsigned_q4.pprof`](../benchmarks/gemma4-gap/audit/go_prefill_retained_unsigned_q4.pprof) and [`go_prefill_retained_unsigned_q4_profile.log`](../benchmarks/gemma4-gap/audit/go_prefill_retained_unsigned_q4_profile.log).

Applying that 81.85% share to the accepted 48.875 tok/s median gives an Amdahl estimate. The current prompt takes 2.5371 seconds, of which approximately 2.0766 seconds are in the tile and 0.4605 seconds elsewhere. Matching the 1.3592-second oracle while leaving the remainder fixed gives the tile 0.8987 seconds: a 2.31 times kernel speed-up, or a 56.7% tile-time reduction. Reaching the 89.405 tok/s 98% gate still requires approximately 2.24 times. This is an estimate because sampled CPU share is not wall-clock decomposition, but it shows why another small scheduling or allocation change cannot close the gap.

## Structural source of the gap

The retained Go kernel and b607's AVX2 prefill kernel perform the same number of scalar output dot products, but organise them differently:

| Property | Retained Go path | llama.cpp b607 AVX2 prefill |
|---|---|---|
| Main output tile | one weight row by eight tokens | eight weight rows by sixteen tokens; four-token tail |
| Weight layout | ordinary 18-byte Q4_0 blocks | eight-row `block_q4_0x8`, 144 bytes |
| Activation layout | eight-token SoA, 288 bytes per QK block | four-row `block_q8_0x4`, 136 bytes |
| YMM lane meaning | eight K-lane FP32 partials for one output | eight independent weight-row outputs |
| Per-block reduction | eight int32 lane dots become eight persistent FP32 FMAs | one complete 32-element integer dot becomes one FP32 FMA per output |
| Final reduction | legacy ordered horizontal sum | outputs are already scalar lanes |

For the 124-token prompt, the minimum input payload makes the reuse difference concrete. Go uses fifteen full eight-token tiles plus one four-token tail. Across eight weight rows that is 38,016 bytes per QK block for 992 output contributions, or 38.32 bytes per output. llama.cpp uses seven sixteen-token supertiles plus three four-token tails: 5,656 bytes for the same 992 contributions, or 5.70 bytes per output. This is a source-layout lower bound rather than measured cache traffic, but it is a 6.7 times arithmetic-intensity advantage. Most of it comes from loading each packed Q8 activation once for eight weight rows instead of once per weight row. Q4 payload and decode are also reused across sixteen tokens rather than eight for most of the prompt.

This reuse is enabled by llama.cpp's reduction topology. Its vector lanes are separate outputs, so a repacked Q4 decode and a packed Q8 load feed a much wider output tile. Go's exact contract assigns a complete YMM register to every output so that all eight K-lane FP32 states survive every block. AVX2 has only sixteen YMM registers: eight persistent outputs plus unpack, scale, correction and dot temporaries already consume the register file. Ordinary row repacking therefore cannot recover llama.cpp's reuse by itself. Eight normal Q4_0 blocks and llama.cpp's eight-row packed tile both occupy 144 bytes, so keeping both layouts would duplicate weight storage rather than compress it. For this model, duplicating all 342 Q4_0 tensors would add 2,219,212,800 bytes (2.066803 GiB).

The dominant retained prefill gap is consequently the exact kernel's output/reduction orientation -- especially repeated Q8 activation loading across weight rows -- rather than generic Go runtime overhead, activation quantisation or worker scheduling. The existing random probe showing 877 differences in 1,000 cases explains why llama.cpp's more intense complete-block reduction cannot simply replace it.

## Lane-transposed exact experiment -- rejected

The bounded lane-transposed four-weight-row/two-token microtile was implemented behind an experimental, non-production entry point. Its block-major Q4 panel contains four FP16 scales and four 16-byte nibble payloads; its Q8 panel contains two FP32 scales and two 32-byte payloads. Both are 72 bytes, so the test-only packers are size-preserving. Logical vector lanes are token-major (`t0r0`--`t0r3`, then `t1r0`--`t1r3`). Eight YMM accumulators each retain one legacy K lane across those eight outputs, and a final exact 8-by-8 transpose restores one eight-lane vector per output before the unchanged reduction.

This preserves the intended 144 source bytes for eight outputs, or 18 bytes per QK block/output -- 52.9% below the retained tile's 38.25-byte full-tile lower bound. It also proves that payload reduction alone is insufficient under the exact contract. AVX-VNNI naturally produces eight K-lane partials for one row/token pair. Reorienting those results requires an 8-by-8 transpose every block while all eight accumulators remain live, plus the final transpose. On AVX2 the required inserts and scratch traffic cost much more than the saved input loads.

One hundred deterministic random cases spanning 1--80 blocks matched the portable reference bit-for-bit for all 64 pre-reduction FP32 lane states, all eight final outputs and the existing output-major kernel. Zero-block and short-buffer edges were also checked. The implementation is therefore exact but not competitive.

Pinned `taskset -c 0-5`, `GOMAXPROCS=6`, five-sample medians were:

| 80-block work | Retained 1-row/8-token | Output-major 4-row/2-token | Lane-transposed 4-row/2-token |
|---|---:|---:|---:|
| Single eight-output tile | 615.5 ns | 1,545 ns | 3,574 ns |
| Synthetic 128-row/124-token projection | 1.876609 ms | 3.151859 ms | 6.285534 ms |

The lane-transposed column uses the optimistic assembly-only path: it performs the final lane transpose but excludes the Go final reduction as well as packing. The complete experimental wrapper projection median was slower again at 7.805817 ms. Even this favourable projection ceiling achieved only 0.299 times retained speed -- 3.35 times slower -- rather than the required 2.24 times speed-up; it would need a further 7.50 times improvement to cross the gate. Despite substantial host variance, the optimistic candidate's fastest projection sample (4.454312 ms) was still 1.98 times the retained path's slowest sample (2.253707 ms), so the rejection is unambiguous. Including packing and reduction cannot recover the deficit. Evidence is in [`go_q4_lane_transposed_exact.log`](../benchmarks/gemma4-gap/audit/go_q4_lane_transposed_exact.log), [`go_q4_lane_transposed_tile_bench.log`](../benchmarks/gemma4-gap/audit/go_q4_lane_transposed_tile_bench.log) and [`go_q4_lane_transposed_projection_bench.log`](../benchmarks/gemma4-gap/audit/go_q4_lane_transposed_projection_bench.log).

The candidate is not integrated into matrix projection, transient packing, complete-request validation or the accepted throughput table. The retained one-row/eight-token dynamic-correction kernel remains the production path. This closes the bounded experiment without duplicating Q4 model storage or relaxing exactness.

## Direct b607 topology experiment -- rejected

A second quarantined experiment deliberately relaxed legacy reduction exactness and ported the arithmetic topology from llama.cpp revision `065d9d50152486590c09b31627ecaf76ceba39dd`. Its `block_q4_0x8` panel is byte-for-byte 144 bytes: eight FP16 scales followed by 128 Q4 bytes interleaved in eight-byte row chunks and XORed with `0x88`. Its `block_q8_0x4` panel is 136 bytes: four FP16 scales followed by 128 Q8 bytes interleaved in four-row, eight-byte chunks. Destination-writing packers are size-checked and pad incomplete eight-row or four-token tails with zero scales.

The portable reference and AVX2/AVX-VNNI implementation complete each 32-element integer dot before one FP32 FMA per output and block. The C intrinsic kernel is isolated in `loader/gguf/llamaq4`; the ordinary Go package supplies only an unexported experimental packed-panel entry point. Whole-projection orchestration processes four Q8_0x4 panels as a 16-token supertile and copies only logical rows and tokens from padded tails. The retained production dispatcher is unchanged.

Byte-layout tests cover both packed structures. Deterministic AVX-VNNI comparisons for every block count from 1 through 80 match the portable topology reference bit-for-bit, and a 13-row/19-token projection verifies the 16-token supertile plus both tails. The explicit non-exactness probe diverged from legacy Go in 24 of 32 outputs for its fixed 80-block input; this is expected because complete integer block reduction replaces the eight persistent FP32 K-lane sequences.

Pinned five-sample medians (`taskset -c 0-5`, `GOMAXPROCS=6`, one-second benchmark windows) were:

| Equal work or projection | Retained | Direct b607 | Relative to retained |
|---|---:|---:|---:|
| 80-block, 32-output equal-work tile | 2,756 ns | 2,976 ns | 0.926× |
| 128-row/124-token projection, prepacked | 1.405774 ms | 1.525595 ms | 0.921× |
| Projection, activation packing included | 1.405774 ms | 2.284316 ms | 0.615× |
| Projection, all packing included | 1.405774 ms | 2.208833 ms | 0.636× |

The candidate therefore misses the mandatory 2.24× projection gate even before packing, and becomes slower once the required activation packing is counted. The plan stops at that gate: there is no model integration, permanent or cached second weight representation, 124+48 state/ID run, or oracle-throughput promotion run. This is an explicit semantics rejection on performance rather than a numerical failure of the intended llama topology. Raw evidence is in [`go_q4_llama_b607_tile_bench.log`](../benchmarks/gemma4-gap/audit/go_q4_llama_b607_tile_bench.log) and [`go_q4_llama_b607_projection_bench.log`](../benchmarks/gemma4-gap/audit/go_q4_llama_b607_projection_bench.log).

## Reproduction and validation

The CPU oracle requires the CUDA-disabled b607 audit build and the F32-KV source defaults captured by `benchmarks/gemma4-gap/audit/llama-b607-exact-cpu.patch`:

```bash
MODEL="$PWD/models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf"
taskset -c 0-5 /workspace/tmp/llama-b607-cpu/build-cpu/bin/llama-bench \
  -m "$MODEL" -p 0 -n 0 -pg 124,47 -r 3 -t 6 -ngl 0 -fa auto -o json
```

The Go side uses the same pinned CPUs and phase boundaries:

```bash
mkdir -p .gotmp
taskset -c 0-5 env \
  GOMAXPROCS=6 GOTMPDIR="$PWD/.gotmp" \
  GO_PHERENCE_GEMMA4_GAP_REAL=1 \
  GO_PHERENCE_GEMMA4_MAIN="$MODEL" \
  go test ./model -run '^TestGemma4RealCPUGap124x48$' \
  -count=3 -v -timeout=10m
```

The retained checkpoint passes focused loader/model tests, focused vet, serialised race tests, exact real-model trajectory checks, `git diff --check`, and Linux arm64/riscv64 compile-only gates. Repository-wide `go test ./...` and `go vet ./...` remain blocked by unrelated pre-existing SpacemiT, diffusion, assembly, unsafe-pointer and Vulkan failures; those failures are not evidence against this CPU checkpoint. Hardware counters are unavailable because `perf_event_paranoid=4`, so phase-separated Go profiles and pinned alternating measurements are the reproducible fallback.
