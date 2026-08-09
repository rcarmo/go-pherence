# Fused 8x16 projection batch

## Mechanism

The corrected C intrinsic path adds `go_llama_q4_0_q8_0_8x16`. For each QK block it:

1. loads one 144-byte `block_q4_0x8` panel;
2. builds the row permutations and decodes its nibbles once;
3. consumes four block-consecutive `block_q8_0x4` panels; and
4. updates 16 token accumulators, each containing eight independent row outputs.

`go_llama_q4_0_q8_0_projection` uses this function for every complete four-panel group and retains the tested 8x4 function for one to three trailing panels. The 124-token synthetic projection therefore uses seven fused 16-token groups and three 4-token tails.

This corrects the invalid serial topology described in `b607-direct-benchmark-audit.md`. Production remains unchanged and no additional persistent weight representation exists.

## Unsigned arithmetic refinement

The first fused implementation retained the signed-dot preparation from the 8x4 port. Its compiler output contained eight `VPSHUFB`, 48 `VPSIGNB` and 32 arithmetic `VPDPBUSD` instructions in the static block/panel loop body.

The refined implementation restores the original unsigned Q4 nibble (`0..15`) while decoding and computes:

```text
sum((q - 8) * a) = sum(q * a) - 8 * sum(a)
```

The activation sum is computed once per token and QK block and its correction is shared by all eight weight rows. Compiler output now contains zero `VPSHUFB`, zero `VPSIGNB` and 36 `VPDPBUSD`: 32 output dots plus four activation-sum operations. The four extra dot instructions replace 48 sign operations and eight lookup shuffles while preserving the complete integer block result.

Static evidence:

- `go_q4_llama_b607_fused_8x16_objdump.log` -- initial signed fused kernel;
- `go_q4_llama_b607_fused_8x16_unsigned_objdump.log` -- unsigned-correction kernel.

The compiler still reserves a 0x5a0-byte stack frame because AVX2 has only 16 architectural YMM registers and the kernel has 16 persistent output vectors plus decoded-Q4 and Q8 temporaries. A future handwritten assembly batch should therefore target spill scheduling rather than merely rewriting the same intrinsics.

## Direct activation preparation and row scheduling

`quantizeQ8_0x4To` writes the fused activation layout directly from row-major F32 input. It is byte-identical to `quantizeQ8_0To` followed by `packQ8_0x4To` across zero-scale blocks, token tails and real 124-by-2560 shapes, but eliminates the intermediate token-major Q8 allocation and second traversal. At long-prefill shapes, approximately four dynamic token-group chunks per worker are claimed atomically. Five background goroutines plus the caller participate at `GOMAXPROCS=6`.

Projection uses the same policy over aligned eight-row groups. Each claim crosses into C once for a row range, then processes all token supertiles. This avoids per-tile cgo calls, makes the caller useful, and attacks the static-tail mechanism found in the scheduler trace. A 129-row/19-token test compares the dynamic six-worker result bit-for-bit with serial C orchestration, including the final padded row group.

## Correctness

`go_q4_llama_b607_fused_unsigned_correctness.log` records:

- byte-exact Q4 and Q8 panel layouts;
- malformed-size rejection;
- the original 8x4 AVX-VNNI/reference comparison for every block count 1--80;
- the fused 8x16 AVX-VNNI/reference comparison for every block count 1--80 and all 128 outputs;
- direct fused-layout quantisation byte equality through a 124-token/2560-wide real shape;
- a 13-row/19-token fused-group and row/token-tail comparison;
- dynamic six-worker versus serial orchestration for 129 rows and 19 tokens; and
- the documented 24/32-output divergence from legacy Go's eight persistent FP32 K-lane reduction.

All passed. The direct-quantisation, fused-reference, tail and dynamic-row tests also pass three repetitions under Go's race detector. `cc -O3 -Wall -Wextra -Werror` accepts the C kernel. Each fused output retains the b607 accumulation order: one complete corrected int32 dot, conversion, scale and one FP32 FMA per QK block.

## Contaminated signed-fusion timing

Two pinned five-sample runs were made before the unsigned refinement. Rui reported additional system load during the runs, so they are not promotion evidence and must not be entered into the accepted throughput table.

The paired run's medians were:

| 128-row/124-token projection | Median | Relative to retained |
|---|---:|---:|
| Retained 1-row/8-token | 698.851 us | 1.000x |
| Signed fused, prepacked | 551.918 us | 1.266x |
| Signed fused, Q8 packing included | 828.200 us | 0.844x |
| Signed fused, all packing included | 1,035.925 us | 0.675x |

Even as contaminated attribution, this is internally coherent with the instruction accounting: removing three repeated Q4 decodes is useful, but Q8 packing can erase the gain. It also shows why a promotion batch needs direct destination-writing Q8 preparation rather than quantise-then-repack.

Raw logs are `go_q4_llama_b607_fused_projection_bench.log` and `go_q4_llama_b607_fused_projection_paired.log`.

## Next gates

When host load is stable, run one pinned benchmark invocation containing:

1. retained `QuantMatrix.ProjectBatchF32To`, including its production quantisation, tile packing and six-worker row projection;
2. fused direct-Q8 F32 projection, including dynamic direct quantisation and dynamic row-range C calls; and
3. the prepacked fused projection only as attribution.

Reject the batch if the complete F32 projection gain is not clear. If it passes, use replacement-only Q4 packing at model load, prove logits, generated IDs and K/V checkpoints, and only then run the frozen 124+48 acceptance protocol. No model integration or throughput timing has yet occurred.
