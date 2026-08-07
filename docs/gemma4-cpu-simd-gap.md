## Gemma4 E4B CPU SIMD gap

This note tracks the awkward part of the Gemma4 E4B CPU work: the model now has an exact layer-batched prefill path, but exactness keeps the Q4_0 kernel far away from llama.cpp's packed GEMM throughput. All measurements below use six Alder Lake cores, `taskset -c 0-5`, `GOMAXPROCS=6`, and the Google QAT Q4_0 model at `models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf`.

The comparison oracle is 447.166 prefill tokens/s and 9.310 decode tokens/s. The 98% gates are therefore 438.223 and 9.124 tokens/s.

## What is retained

Long prefill uses an AVX-VNNI one-weight-row by eight-token kernel. Each output keeps the legacy eight FP32 lane accumulators, including blockwise FMA and final reduction order, so the batched path remains bit-identical to sequential execution. Each eight-token input block now has structure-of-arrays layout: eight contiguous FP32 scales followed by eight contiguous Q8 vectors. One packed scale load and multiply replaces eight scalar scale sequences without changing arithmetic. One hundred randomized 1--80-block comparisons are bit-exact. Five paired hot-kernel runs improved by 12.7--16.5%, and writing quantization into preallocated activation storage reduced a representative batch-124 projection from 141 to 17 allocations and from about 1.09MiB to 0.70MiB per call.

The preceding block-major layout had already improved the alternating end-to-end median from 26.027 to 26.804 tokens/s (2.98%). The structure-of-arrays projection benchmark has a favorable median, but real-model runs remain too sensitive to unrelated host load to claim another stable end-to-end percentage: adjacent samples have moved by 10--30% in either direction while the paired kernel result remains stable. The full 124-token activation, logits, token, and K/V comparison passes exactly, so the layout is retained on the stronger isolated evidence rather than a cherry-picked model sample.

Decode also avoids a 512-byte Q6_K coefficient temporary per block. On AVX-VNNI hosts the retained kernel fuses the complete multi-block GEMV in assembly, uses the existing sixteen Q8_K sums with `VPMADDWD` to form the unsigned-Q6 correction once per block, and expands each eight-byte scale half once before selecting scale pairs with `VPERMD`. Blockwise FP32 multiplication and accumulation remain in their original order. One hundred randomized 1--10-block comparisons are bit-exact against the prior Go loop. Five alternating 2560-element measurements put the fused path at 111.8--214.9ns against 159.0--300.6ns for that loop, a stable 39.9--42.2% throughput gain across frequency states.

The Q4_0 decode kernel first moved to four weight rows per Q8 activation load, retaining four independent eight-lane FP32 accumulators and replacing per-row signed-byte transforms with the unsigned identity `q4*q8 - 8*q8 == (q4-8)*q8`. It now evaluates eight adjacent rows per call and consumes a precomputed per-block `8*q8` correction, so the correction dot is paid once during Q8 quantization rather than once per row group. One hundred randomized 1--80-block comparisons remain bit-exact, including tails that fall back to four or scalar rows. Adjacent alternating 2560-element measurements put the eight-row kernel 2.1--2.9% ahead of two four-row calls; the larger gain comes from removing repeated correction work across all row groups.

The final five-sample real-model decode gate measured 8.166, 9.453, 9.418, 8.447, and 9.294 tokens/s. The 9.294 median is 99.8% of the 9.310 llama.cpp oracle and exceeds the 9.124 acceptance threshold; three individual samples also passed. A subsequent profile-bearing run measured 9.220 tokens/s. Host frequency remains visibly bimodal, so the median rather than a single best sample is the acceptance result.

## What was rejected

The four-weight-row by two-token Q4_0 tile preserves exact reduction order but reduced prefill to 20.53--21.15 tokens/s. Rewriting it around the unsigned-Q4 correction made it only 2--4% slower than the retained token tile in paired microbenchmarks, but it still lost in the real projection path. A two-row by four-token tile went 2.6--4.2% faster in five paired kernel runs by reusing each Q8 load, yet lost four of five end-to-end pairs by 4.9--7.6%; its extra weight passes and scatter overhead outweigh the isolated gain.

Gathering eight activation scales into a packed vector was exact but 1--4% slower in four of five paired runs. The retained structure-of-arrays layout gets the same packed scale arithmetic from one contiguous load instead.

A direct signed-byte `VPDPBSSD` replacement is illegal on this Alder Lake host. `avx_vnni` provides `VPDPBUSD`; signed-byte `VPDPBSSD` requires the newer AVX-VNNI-INT8 extension. The prefill token kernel therefore keeps `VPABSB` plus `VPSIGNB`, followed by unsigned-by-signed `VPDPBUSD`; decode uses the unsigned-Q4 correction described above.

A llama.cpp-style packed reduction is not an exact substitute for the legacy contract. It reduces a whole 32-element block to one integer before FP32 accumulation, while go-pherence maintains eight FP32 partial accumulators until the final reduction. A deterministic random probe differed in 877 of 1000 cases; the first mismatch was 5479.129 versus 5479.1265. Promoting that reduction would require an explicit contract change, not a kernel refactor disguised as one.

Fully unrolling the fused Q6_K block dot also lost. It remained exact across 1000 random blocks and halved the packed-source loads by reusing QL/QH bytes for low and high groups, but increased the median from 27.79ns to 29.49ns per block. The smaller looped kernel has a better instruction footprint on this CPU.

## Where the time goes

A prefill-only CPU profile of the earlier block-major kernel measured 27.767 tokens/s and 14.38 CPU-seconds over 4.47 wall-seconds. `dotQ4_0Q8_0Tokens8VNNI` accounted for 81.08% of CPU samples. Worker scaling was 8.228, 13.792, 18.903, 21.454, and 23.812 tokens/s at 1, 2, 3, 4, and 6 workers respectively, which is the signature of a memory-bound exact kernel rather than a missing goroutine. A phase-specific structure-of-arrays profile attributes 83.38% of samples to `dotQ4_0Q8_0Tokens8SoAVNNI`; repeated Q8 vector loads are its dominant sampled instructions. That run overlapped unrelated host work and is retained for attribution, not throughput.

The final decode-only profile measured 9.220 tokens/s and 20.73 CPU-seconds over 5.21 wall-seconds. `dotQ4_0Q8_0x8VNNI` accounted for 74.19% of samples and the fused Q6_K GEMV for 19.44%; quantization, dense rows, activation functions, and runtime overhead were each below 1.4%. Decode has crossed its acceptance gate, and Q4_0 remains the only meaningful target if additional headroom is required.

AVX2 has sixteen YMM registers. One exact Q4_0 output needs one full YMM accumulator to retain its eight FP32 partial sums, leaving room for only eight outputs plus unpack, scale, and dot-product temporaries. llama.cpp gets its arithmetic intensity by changing the reduction contract and tiling many rows and tokens together. Closing the remaining prefill gap therefore needs either a wider canonical reduction contract with new frozen outputs, or another way to reproduce the eight-lane result without rereading weights and activations. Ordinary row repacking does not remove that register constraint.

## Reproduction and gates

```bash
mkdir -p .gotmp
export GOTMPDIR=$PWD/.gotmp GOMAXPROCS=6
export GO_PHERENCE_GEMMA4_MAIN=$PWD/models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf

GO_PHERENCE_GEMMA4_GAP_REAL=1 \
  taskset -c 0-5 go test ./model \
  -run '^TestGemma4RealCPUGap124x48$' -count=5 -v

GO_PHERENCE_GEMMA4_GAP_REAL=1 \
GO_PHERENCE_GEMMA4_PREFILL_ONLY=1 \
GO_PHERENCE_GEMMA4_PREFILL_CPU_PROFILE=$PWD/tmp/prefill.pprof \
  taskset -c 0-5 go test ./model \
  -run '^TestGemma4RealCPUGap124x48$' -count=1 -v

GO_PHERENCE_GEMMA4_PREFILL_CANDIDATE_REAL_LONG=1 \
  taskset -c 0-5 go test ./model \
  -run '^TestGemma4LegacyPrefillCandidate124RealParity$' -count=1 -v

taskset -c 0-5 go test ./loader/gguf ./model -count=1
taskset -c 0-5 go test -race ./loader/gguf ./model -count=1
```

The regular loader/model suite, race suite, exact real-model 124-token comparison, `git diff --check`, and Linux arm64/riscv64 compile-only checks pass. Hardware counter attribution is unavailable on this machine because `perf_event_paranoid=4`; phase-separated Go CPU profiles and alternating A/B runs are the reproducible fallback.
