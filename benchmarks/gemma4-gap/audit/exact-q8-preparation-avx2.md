# Exact Q8 preparation with AVX2

## Status

Accepted and promoted with direct-to-SoA preparation after independent row-block attribution. The candidate was correctness-gated in isolated worktree `/tmp/go-pherence-direct-q8` at baseline `1c1ff55d`; the original reproducible diff is [`exact-q8-preparation-avx2.patch`](exact-q8-preparation-avx2.patch).

## Mechanism

The compiler emits scalar loops for `quantizeQ8_0BlockTo`. Per 32-value block it converts each value to float64 for `math.Abs`, and the inlined `math.Round` path executes scalar exponent, mask, shift and conversion logic before clamping. The saved disassembly is [`exact-q8-preparation-scalar.objdump`](exact-q8-preparation-scalar.objdump).

The candidate adds an amd64 AVX2 block quantiser:

- four unaligned YMM loads consume 32 FP32 values;
- ordered masks clear NaNs before packed absolute maxima, preserving the scalar loop's ignore-NaN behaviour, followed by a horizontal reduction;
- scale and inverse retain scalar FP32 division;
- each packed product is widened to float64 before signed `0.5` addition, preserving `math.Round(float64(float32(v*id)))` exactly rather than incorrectly rounding near-half float32 values;
- packed truncation, int32 clamps and saturating packs emit 32 Q8 bytes with two stores;
- the unrounded scale is returned to Go, where the unchanged `half.F32ToF16`/`half.F16ToF32` tie-up conversion produces the stored scale.

The SIMD body has 218 static disassembly instructions and no per-element loop. Static line count alone is not a speed claim, but the scalar body has two 32-iteration loops; its common quantisation iteration includes the inlined float64 floor/round path. Saved candidate disassembly is [`exact-q8-preparation-avx2.objdump`](exact-q8-preparation-avx2.objdump).

Non-amd64 uses the unchanged scalar helper.

## Exactness work

An initial packed `product + copysign(0.5)` float32 implementation was correctly rejected by a next-representable-value test: `-0.49999997 + -0.5` rounds to float32 `-1`, whereas `math.Round(float64(-0.49999997))` returns zero. The accepted assembly widens products to float64 before adding `0.5`.

The final exact test compares SIMD and the retained scalar helper over:

- zero, positive/negative zero-scale and smallest-subnormal blocks;
- NaN in every one of the 32 input positions and positive/negative infinity behaviour;
- exact half values and both adjacent float32 values for every `n+0.5`, `n=-126..126`;
- 10,000 deterministic random blocks over varied magnitudes;
- direct tiled shapes `(64,32)`, `(65,256)` and real `(124,2560)`, including the four-token tail;
- projection-oracle outputs.

Saved gate output is [`exact-q8-preparation-avx2-correctness.log`](exact-q8-preparation-avx2-correctness.log):

```text
GOMAXPROCS=6 go test ./loader/gguf -count=1                         ok
focused SIMD/direct/oracle tests -count=20                          ok
focused race -count=3                                               ok
go vet ./loader/gguf                                                ok
CGO_ENABLED=0 GOOS=linux GOARCH={arm64,riscv64} go test -c          ok
```

## Coherent preparation batch

At the real 124-by-2560 shape, direct-to-SoA preparation removes 345,600 transient bytes and one 691,200-byte copy traversal. AVX2 then attacks the arithmetic that remains; quantisation previously accounted for 4.07% of the retained CPU profile. Even an ideal removal of that profile share cannot close the final gate, so this is an incremental candidate to layer only after independent row-block attribution.

The clean three-way comparison used the independently accepted row-block traversal in all variants:

1. retained quantise-then-pack preparation;
2. exact direct-to-SoA preparation with the scalar block helper;
3. direct-to-SoA plus exact AVX2 quantisation.

Medians were 10.107, 10.141 and 9.938 ms/op respectively. Direct-scalar was flat while halving transient allocation; AVX2 reduced the median by 1.68%. A longer five-pair, three-second comparison then put retained and AVX2 medians at **10.070 and 9.707 ms/op**, a **3.60% reduction**. AVX2 won every pair. Two pairs contained a brief Jupyter process at one boundary; excluding them leaves three clean wins and a 2.43% median reduction. Evidence is [`go_q4_row_block_direct_q8_three_way.log`](go_q4_row_block_direct_q8_three_way.log), [`go_q4_row_block_direct_avx2_long_pairs.log`](go_q4_row_block_direct_avx2_long_pairs.log) and their corresponding load logs.

The promoted exact 124+48 gate matched all 48 frozen IDs. Three clean order-balanced full-request pairs produced baseline medians of **42.976 prompt / 9.652 decode tok/s** and candidate medians of **49.152 prompt / 10.189 decode tok/s**: **+14.4% prompt and +5.6% decode** in the paired environment. Candidate prompt efficiency against the 91.2296 tok/s oracle is 53.9%; decode efficiency against 10.5265 tok/s is 96.8%. The only external processes reaching 20% occurred between or after requests. Full evidence is [`go_final_exact_batch_full_request_paired_r3.log`](go_final_exact_batch_full_request_paired_r3.log) and [`go_final_exact_batch_full_request_paired_r3_load.log`](go_final_exact_batch_full_request_paired_r3_load.log).
