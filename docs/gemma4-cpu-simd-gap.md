## Gemma4 E4B CPU SIMD gap

This note tracks the awkward part of the Gemma4 E4B CPU work: the model now has an exact layer-batched prefill path, but exactness keeps the Q4_0 kernel far away from llama.cpp's packed GEMM throughput. All measurements below use six Alder Lake cores, `taskset -c 0-5`, `GOMAXPROCS=6`, and the Google QAT Q4_0 model at `models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf`.

The comparison oracle is 447.166 prefill tokens/s and 9.310 decode tokens/s. The 98% gates are therefore 438.223 and 9.124 tokens/s.

## What is retained

Long prefill uses an AVX-VNNI one-weight-row by eight-token kernel. Each output keeps the legacy eight FP32 lane accumulators, including blockwise FMA and final reduction order, so the batched path remains bit-identical to sequential execution. Q8_0 activation blocks are transposed within each eight-token tile from token-major to block-major order; this replaces eight activation streams roughly 2.9KiB apart with contiguous block reads without changing arithmetic.

An alternating end-to-end A/B run measured token-major prefill at 27.997, 26.027, and 25.224 tokens/s, against 26.361, 27.181, and 26.804 tokens/s for block-major tiles. The medians are 26.027 and 26.804 tokens/s, a 2.98% gain. Host frequency and thermal behaviour are noisy enough that this is a modest result, but the full 124-token activation, logits, token, and K/V comparison passes exactly.

Decode also avoids a 512-byte Q6_K coefficient temporary per block. The fused AVX2 kernel expands Q6 values, applies the signed scales, consumes the matching Q8_K bytes, and accumulates the same eight integer lanes in one pass. Its 2560-element microbenchmark is approximately 304ns, while the previous expand-then-dot path measured 354--716ns under the same run.

The Q4_0 decode kernel now evaluates four weight rows per Q8 activation load. On AVX-VNNI hosts it keeps four independent eight-lane FP32 accumulators, so blockwise FMA and final reduction remain exact, but replaces per-row signed-byte transforms with an unsigned identity: `q4*q8 - 8*q8 == (q4-8)*q8`. One `VPDPBUSD` computes the `8*q8` lane correction per block and each row subtracts it from its unsigned Q4 dot. The four-row microbenchmark median fell from 602.7ns with AVX2 to 421.3ns with the signed VNNI form; the unsigned correction then reduced the complete 4096x2560 four-row GEMV median from 112.345us to 108.399us.

Alternating real-model A/B samples were run while host frequency drifted badly enough to halve absolute throughput. AVX2 decode measured 6.704 and 4.258 tokens/s; the exact VNNI kernel measured 8.616 and 5.409 tokens/s in the adjacent runs, a stable 27--29% relative gain. The uncontaminated 98% gate is still pending.

## What was rejected

The four-weight-row by two-token Q4_0 tile preserves exact reduction order but reduced prefill to 20.53--21.15 tokens/s. The retained one-row by eight-token layout measured 23.08--29.21 tokens/s in the same class of end-to-end runs.

A direct signed-byte `VPDPBSSD` replacement is illegal on this Alder Lake host. `avx_vnni` provides `VPDPBUSD`; signed-byte `VPDPBSSD` requires the newer AVX-VNNI-INT8 extension. The prefill token kernel therefore keeps `VPABSB` plus `VPSIGNB`, followed by unsigned-by-signed `VPDPBUSD`; decode uses the unsigned-Q4 correction described above.

A llama.cpp-style packed reduction is not an exact substitute for the legacy contract. It reduces a whole 32-element block to one integer before FP32 accumulation, while go-pherence maintains eight FP32 partial accumulators until the final reduction. A deterministic random probe differed in 877 of 1000 cases; the first mismatch was 5479.129 versus 5479.1265. Promoting that reduction would require an explicit contract change, not a kernel refactor disguised as one.

Fully unrolling the fused Q6_K block dot also lost. It remained exact across 1000 random blocks and halved the packed-source loads by reusing QL/QH bytes for low and high groups, but increased the median from 27.79ns to 29.49ns per block. The smaller looped kernel has a better instruction footprint on this CPU.

## Where the time goes

A prefill-only CPU profile measured 27.767 tokens/s and 14.38 CPU-seconds over 4.47 wall-seconds. `dotQ4_0Q8_0Tokens8VNNI` accounted for 81.08% of CPU samples. Worker scaling was 8.228, 13.792, 18.903, 21.454, and 23.812 tokens/s at 1, 2, 3, 4, and 6 workers respectively, which is the signature of a memory-bound exact kernel rather than a missing goroutine.

The corresponding decode-only profile measured 7.903 tokens/s before Q6 fusion. Q4_0 row projection accounted for 70.48% of samples and Q6_K for 23.58%. After fusion, the Q6 share fell to 22.09%; Q4_0 row projection is now the main decode target at roughly 71%.

AVX2 has sixteen YMM registers. One exact Q4_0 output needs one full YMM accumulator to retain its eight FP32 partial sums, leaving room for only eight outputs plus unpack, scale, and dot-product temporaries. llama.cpp gets its arithmetic intensity by changing the reduction contract and tiling many rows and tokens together. Closing the remaining prefill gap therefore needs either a wider canonical reduction contract with new frozen outputs, or another way to reproduce the eight-lane result without rereading weights and activations. Ordinary row repacking does not remove that register constraint.

## Reproduction and gates

```bash
mkdir -p .gotmp
export GOTMPDIR=$PWD/.gotmp GOMAXPROCS=6
export GO_PHERENCE_GEMMA4_MAIN=$PWD/models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf

GO_PHERENCE_GEMMA4_GAP_REAL=1 \
  taskset -c 0-5 go test ./model \
  -run '^TestGemma4RealCPUGap124x48$' -count=3 -v

GO_PHERENCE_GEMMA4_PREFILL_CANDIDATE_REAL_LONG=1 \
  taskset -c 0-5 go test ./model \
  -run '^TestGemma4LegacyPrefillCandidate124RealParity$' -count=1 -v

taskset -c 0-5 go test ./loader/gguf ./model -count=1
taskset -c 0-5 go test -race ./loader/gguf ./model -count=1
```

The regular loader/model suite, race suite, exact real-model 124-token comparison, `git diff --check`, and Linux arm64/riscv64 compile-only checks pass. Hardware counter attribution is unavailable on this machine because `perf_event_paranoid=4`; phase-separated Go CPU profiles and alternating A/B runs are the reproducible fallback.
