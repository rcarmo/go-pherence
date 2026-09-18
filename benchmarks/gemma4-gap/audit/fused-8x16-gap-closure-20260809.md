# Fused 8x16 prompt-gap closure assessment -- 2026-08-09

## Budget

The promoted fused path runs the 124-token prompt at 64.799044 tok/s, or 1.913608 s. Reaching 89.405 tok/s requires 1.386947 s, leaving 526.661 ms to remove from one request -- 27.52% of total prompt time.

The profile is a parallel CPU profile, so its 81.08% flat share cannot be treated as a wall-time fraction without qualification. The historical sample-share Amdahl calculation assigns 1.551554 s to native execution and 362.055 ms to everything else; on that model the hotspot must fall to 1.024892 s, a 1.514x speed-up or 33.94% reduction.

The worker-span estimate is stricter. Six workers accumulated 6.13 CPU-seconds in native code, giving a minimum balanced span of 1.022 s, while the caller-side projection stack sampled about 1.11 s. A native span in the 1.02--1.11 s range must fall by the same 526.661 ms request gap, requiring approximately 1.90--2.06x. The 1.514x result is therefore an optimistic sample-share bound; 1.9x is the sensible projection gate until explicit wall-span instrumentation replaces the estimate.

Using the 1.11 s caller-side projection estimate gives the following end-to-end rates:

| Projection speed-up | Implied prompt rate |
| ---: | ---: |
| 1.10x | 68.406 tok/s |
| 1.20x | 71.734 tok/s |
| 1.30x | 74.814 tok/s |
| 1.50x | 80.331 tok/s |
| 1.75x | 86.237 tok/s |
| 1.90x | 89.349 tok/s |
| 2.00x | 91.270 tok/s |

The 2.00x projection result is almost exactly the frozen 91.229561 tok/s llama.cpp oracle. Minor wrapper changes cannot supply that factor.

## What the profile says

`runtime.cgocall` owns 6.13 s, or 81.08%, of aggregate samples summed across the six native workers. Those samples sit below `ProjectQ4_0x8Q8_0x4RowsVNNI`; they are time spent inside C, not a measurement of the language-crossing instruction itself. Row-range calls already amortise the boundary over many row groups, dynamic chunks use five background workers plus the caller, and the promoted scheduler removed the measured static tail.

Removing cgo, changing the atomic claim size, or moving the same loops into Go cannot erase those samples. A scheduling change now needs proof of idle cores or a shorter native critical path, rather than a large flat `runtime.cgocall` number.

## The native instruction budget

The unsigned 8x16 function reserves a 0x5a0-byte frame. Its static disassembly contains 76 vector moves to or from the stack -- 48 loads and 28 stores -- alongside 36 `VPDPBUSD` sites and four FP32 FMA sites.

The block loop is more revealing because the four-panel body is dynamic:

| Region per QK block | Executions | Instructions | stack references | `VPDPBUSD` | FMA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 load/decode | 1 | 83 | 19 | 0 | 0 |
| one four-token panel | 4 | 210 | 40 | 36 | 4 |
| effective total | -- | 923 | 179 | 144 | 16 |

The 144 dot instructions comprise 128 output dots -- the arithmetic minimum for 128 outputs over 32 K values with 256-bit VNNI -- plus 16 activation-sum dots. The remaining 779 instructions and 179 stack references are the available optimisation surface. This is a support-instruction and data-movement problem; replacing VNNI arithmetic with another expression does not provide a 2x result by itself.

The C row driver adds another avoidable layer. It calls the kernel into a 128-float local tile and scatters that tile into the strided final output. The compiled `projection_rows` function is 818 instructions with separate full-row and tail paths. A stride-aware kernel can store its 16 final YMM results directly and remove the local tile plus scalar scatter, although this is incremental rather than gap-closing.

The 124-token shape also leaves 12 tokens on three calls to the older signed 8x4 tail. That function has a 0x2c8-byte frame, 79 stack references, 32 dot sites, 48 `VPSIGNB`, and eight `VPSHUFB` sites. An unsigned fused 8x12 tail removes redundant Q4 decode and signed-byte preparation, but it touches only 9.68% of prompt tokens.

## Ranked mechanisms

### 1. Compare or port the complete frozen llama.cpp GEMM path

This is the only mechanism with measured end-to-end precedent for the required scale: the same host and model reach 91.229561 tok/s. The next benchmark should compare equal packed inputs and outputs for the current native row driver and the frozen llama.cpp GEMM driver, with `perf stat` or equivalent counters for cycles, instructions, L1/L2 traffic and branch misses. Comparing only source-level inner tiles is insufficient because output placement, panel traversal and thread ownership are part of the cost.

If the frozen driver is close to 2x on an equal projection, port its kernel and traversal behind the existing CPU/build dispatch. Replacement-only Q4 storage, tied embedding/head ownership, K/V aliases and no-cgo fallback semantics remain unchanged. If both native drivers are close on an equal projection, the 91.23 tok/s oracle comes from work outside this kernel and the profile must be recollected with native symbols before another rewrite.

### 2. Build a spill-reduced native kernel around a predeclared 1.9x gate

A handwritten AVX2/AVX-VNNI kernel is justified only if its schedule cuts the 923-instruction/179-stack-reference block substantially. The first candidates worth assembling are an 8x8 two-panel schedule and a four-row/paired-token lane layout, not a transcription of the current intrinsics.

The 8x8 variant has a real penalty: processing 16 tokens needs two Q4 decodes, raising the equal-work static estimate from 923 to roughly 1,006 instructions before spill savings. It should be rejected unless the assembled equal-work body removes enough stack traffic to overcome that extra decode and the direct prepacked projection approaches 1.9x. Merely reducing the 0x5a0 frame is not a performance mechanism.

A paired-token lane layout can reduce live output vectors from 16 to eight by placing two tokens across the lanes of one accumulator. It requires a different replacement-only weight/activation packing and must prove that its extra broadcasts or lane permutations cost less than the spills it removes. Historical exact output-major kernels do not validate this non-exact layout, but they are a warning against assuming register fit means speed.

### 3. Precompute compact activation corrections in direct Q8 preparation

Each 136-byte Q8_0x4 block can carry four signed 16-bit values holding `8 * sum(qs)` for its four tokens, growing to 144 bytes (+5.88%). The direct quantiser can accumulate these sums while writing Q8 bytes; the kernel can replace four activation-sum `VPDPBUSD`, the horizontal add and per-token shifts with one compact load/sign-extension and broadcasts or permutations.

This removes 16 of 144 dynamic VNNI dots per 8x16 QK block -- 11.11% of dot instructions -- without the 256-byte lane-correction expansion that made the earlier exact experiment regress. A 5--10% native gain is plausible. It cannot close a 47.45% projection-time reduction alone.

### 4. Remove native scratch/scatter and fuse the 12-token tail

Make the kernel accept the final output base and row stride, storing complete row groups directly. Add one unsigned 8x12 path for the fixed 124-token remainder, retaining the tested 8x4 and scalar fallbacks for general tails. These changes remove concrete instructions and data movement, and they should be tested as one native-plumbing batch.

Even an optimistic 5% scatter gain plus a 25% improvement over the 9.68% tail contributes only high-single-digit projection improvement. This is useful cleanup before judging a new kernel, not a target claim.

### 5. Reuse activation preparation across projections with the same input

Q/K/V consume one activation and gate/up consume another shared activation. Quantising once and projecting several packed matrices removes duplicate scalar `math.Round`, FP16 scale conversion, allocation and traversal. Gate/up dominate the output-row count, so this is the useful case; simple concatenation without Q8 reuse inside the native loop only removes preparation and call setup.

The profile gives direct Q8 preparation about 60 ms on the caller path. Perfectly deleting all of it would still save only 3.1% of prompt time, so activation sharing belongs after the native discriminator. A dual-weight kernel that also shares transformed Q8 operands may be more valuable, but then it is another native topology and should be judged by the same equal-work gate.

## Low-priority or rejected directions

* cgo-call elimination and smaller row claims rearrange an already amortised, caller-participating schedule. Require a trace showing idle cores before changing them.
* Wider 8x32 fusion increases live accumulators and spill pressure. It amortises an 83-instruction Q4 decode while making the measured problem worse.
* A plain 8x8 split is not automatically spill reduction -- it repeats Q4 decode and still needs decoded-Q4 temporaries. Its assembly and equal-work timing decide.
* Fully expanding Q4 nibbles at load time trades decode instructions for a much larger permanent weight working set. Replacement-only ownership avoids duplication but not the bandwidth and memory-capacity cost.
* Quantiser-only AVX2 work and gate/up Q8 sharing are bounded by the small preparation share. They may improve a coherent batch but cannot be presented as closure mechanisms.

## Next gate

The smallest useful batch is compact Q8 correction + direct strided stores + unsigned 8x12 tail. It must pass the existing fused reference, token/row tails, real finite request, checkpoint restore and no-cgo/build fallback gates, followed by the pinned complete F32 projection benchmark. It is an incremental batch; no end-to-end target run is warranted unless projection improves clearly.

In parallel, the closure experiment should compile the frozen llama.cpp GEMM driver and the current driver into one equal-input benchmark binary. Promote or port only a native path that approaches 1.9x projection speed before spending another 124+48 run. If the oracle driver does not provide that gain in isolation, the current 81.08% profile needs native-symbol attribution rather than another speculative tile rewrite.
