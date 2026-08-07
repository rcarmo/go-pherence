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

Q8_0 activation quantisation now writes directly into caller-owned storage across contiguous worker spans. The parallel and serial paths produce identical scales and quantised bytes, retain oversized-slice behaviour and pass the race detector. Alternating baseline/candidate runs moved complete prompt throughput from a 30.932 tok/s median to 38.691 tok/s (+25.1%). A subsequent retained three-run check measured 32.227, 38.808 and 39.741 prompt tok/s, giving a 38.808 tok/s median.

Decode retains the fused Q6_K multi-block AVX-VNNI GEMV and the eight-row Q4_0 kernel with a precomputed per-block `8*q8` correction. These remove repeated coefficient expansion and correction work while preserving the legacy blockwise FP32 result. Current generation samples are 8.144, 8.753 and 9.344 eval tok/s, giving an 8.753 eval tok/s median.

| Runtime | Prompt | Generation | Oracle efficiency |
|---|---:|---:|---:|
| CPU-only llama.cpp b607, exact phases | 91.230 tok/s | 10.526 eval tok/s | 100% / 100% |
| Go, retained median | 38.808 tok/s | 8.753 eval tok/s | 42.5% / 83.2% |

Neither phase currently passes its corrected 98% gate. Decode is 16.8% below the oracle; prefill remains 57.5% below it.

## What was rejected

The four-weight-row/two-token Q4_0 tile preserves exact reduction order but reduced prefill to 20.53--21.15 tok/s. Rewriting it around the unsigned-Q4 correction narrowed the isolated-kernel loss, but it still lost in the real projection path. A two-row/four-token tile improved its isolated kernel by 2.6--4.2%, then lost four of five end-to-end pairs by 4.9--7.6% because extra weight passes and scatter overhead outweighed activation reuse.

Splitting the retained eight-token tile into two four-token passes freed four YMM registers and allowed two VNNI dependency chains to overlap. It remained exact, but longer pinned samples regressed from an 812 ns median to 1,009 ns because every Q4 block had to be loaded and decoded twice.

Gathering eight activation scales into a packed vector was exact but 1--4% slower in four of five paired runs. The retained structure-of-arrays layout obtains the same packed scale arithmetic from one contiguous load. Flat-interleaved activation and persistent-worker variants also regressed the complete request.

A direct signed-byte `VPDPBSSD` replacement is unavailable on this Alder Lake host. `avx_vnni` provides `VPDPBUSD`; signed-byte `VPDPBSSD` requires AVX-VNNI-INT8. The prefill kernel therefore keeps `VPABSB` plus `VPSIGNB`, followed by unsigned-by-signed `VPDPBUSD`.

A llama.cpp-style packed reduction is not an exact substitute for the legacy contract. It reduces a complete 32-element block to one integer before FP32 accumulation, while go-pherence keeps eight FP32 partial accumulators until final reduction. A deterministic random probe differed in 877 of 1,000 cases. Fully unrolling the fused Q6_K block dot was exact across 1,000 random blocks, but increased its median from 27.79 ns to 29.49 ns.

## Where the time goes

The latest phase-specific profile attributes 83.48% of flat CPU samples to `dotQ4_0Q8_0Tokens8SoAVNNI`. Eliminating every remaining non-Q4 sample would move 38.8 tok/s only to roughly 46.5 tok/s. Closing the prompt gap therefore requires a materially faster exact Q4 matrix tile rather than more scheduling or allocation work.

AVX2 has sixteen YMM registers. One exact output occupies one full YMM accumulator to preserve its eight FP32 partial sums, leaving room for eight outputs plus unpack, scale and dot-product temporaries. llama.cpp obtains greater arithmetic intensity by choosing a different reduction scheme and tiling more rows and tokens together. Ordinary row repacking does not remove this register constraint.

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
