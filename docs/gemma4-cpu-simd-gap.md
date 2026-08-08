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

## Next exact experiment

The next bounded experiment is a lane-transposed four-weight-row/two-token microtile. It differs from the rejected output-major four-row/two-token candidate: eight YMM accumulators would each hold one legacy K lane across eight outputs, rather than one output across eight K lanes. For every block, a size-preserving four-row Q4 panel and a two-token lane-major Q8 panel would produce eight output contributions per VNNI sequence. After the final block, an exact 8-by-8 register transpose would restore one vector per output and invoke the unchanged legacy reduction. Each scalar FP32 lane would see the same blockwise FMA sequence as today; only its register location would change.

The proposed 4-by-2 tile has a 72-byte Q4 panel plus two 36-byte Q8 blocks: 144 source bytes for eight outputs, or 18 bytes per QK block/output. That is 52.9% below the retained tile's 38.25-byte full-tile lower bound and is the best-balanced eight-output rectangle for 18-byte weights and 36-byte activations. It also replaces eight spilled scalar-scale broadcasts with one eight-output outer-product scale vector. It does not change the payload permanently: test-only packers should establish the ceiling before any matrix ownership change is considered.

Validation is deliberately staged:

1. Compare random 1--80-block outputs bit-for-bit with the legacy path, including every one of the eight pre-reduction FP32 lanes.
2. Benchmark equal-work retained, existing output-major 4-by-2 and lane-transposed 4-by-2 kernels with pinned CPUs. The projection-shaped ceiling must reach at least 2.24 times the retained hotspot for the 98% gate, and 2.31 times for oracle parity under the fixed-remainder estimate.
3. Reject the layout before model integration if it cannot meet that requirement. If it does, include transient packing cost and run alternating complete requests with all 48 output IDs unchanged.

This experiment directly tests whether activation reuse can be recovered without relaxing exactness. It is not a commitment to a second persistent Q4 representation.

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
