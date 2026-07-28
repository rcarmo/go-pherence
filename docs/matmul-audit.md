# Matmul optimisation audit

This audit applies the useful design rules from Justine Tunney's [LLaMA Now Goes Faster on CPUs](http://justine.lol/matmul/) to the complete go-pherence multiplication stack.

The article's transferable rules are:

1. Decode (`M=1`) and prompt/prefill (`M>1`) need different kernels.
2. A useful GEMM microkernel computes several output rows and columns at once, retaining multiple accumulators and reusing each loaded operand.
3. Static weights should be packed once into the traversal order required by the microkernel.
4. Cache tiling and register tiling are separate. L1/L2 blocking does not compensate for a one-output inner kernel.
5. Threads should own output tiles supplied by the caller's worker model, rather than spawning work per dot product.
6. Dispatch should depend on datatype, dimensions, ISA, and workload size. Tail kernels are part of the design.

## Highest-priority candidates

| Priority | Kernel family | Current limitation | Recommended work |
|---|---|---|---|
| P0 | GPTQ Q4 CPU batch: `backends/simd/quant/q4.Gemm` | On non-RVV hosts, batched execution is repeated scalar GEMV, usually partitioned by activation row. The riscv64 accelerator already dequantises a weight row once across batch rows. | Add AVX2/NEON `MxN` microkernels (start 2x4/4x4), sharing packed-weight decode across activation rows; pack group metadata; partition output tiles. Keep existing GEMV for decode and the RVV dequant-once path. |
| P0 | MLX affine Q4 CPU batch: `backends/mlx.Gemm` | Explicitly calls validated scalar GEMV once per row; no native batched quant kernel. This affects CPU prompt evaluation for MLX checkpoints. | Implement datatype-native batched kernels over 32-value groups, reusing unpacked nibbles/scales across 2-4 activation rows. Add prepacked output-row tiles and batch/shape dispatch. |
| P0 | NVFP4 CPU batch: `backends/simd/quant/nvfp4.GemmNVFP4` | Generic non-RVV path repeats GEMV; decode of packed FP4 and two-level scales is repeated per activation row. The riscv64 accelerator already reuses each dequantised weight row across the batch. | Add AVX2/NEON multi-row kernels that decode each packed weight group once and accumulate several prompt rows. Preserve single-row GEMV and the RVV path. |
| P0 | GGUF quant row and batch kernels: `loader/gguf/qdot.go`, `loader/gguf/quant_project.go`, `loader/gguf/q4k_gemm_8x8.go`, `model/gguf_quant_rvv.go` | Q4_0 and Q6_K expose generic row kernels; Q4_K, Q6_K, Q5_0 and Q8_0 also have batch project paths, and Q4_K has an 8x8 tile used by DiffusionGemma. Coverage is fragmented by format/caller, while several generic and tail paths still unpack metadata per row in Go. | Extend x4/x8 output-row and tiled batch coverage consistently across formats and generic LLM/MTP callers; share Q8 activation loads; unpack scales/mins in SIMD registers; retain the existing Q4_K 8x8 path as the baseline. |
| P0 | BF16 prompt GEMM: `GemmRowsBF16Parallel` and MOSS adaptor | Repeats BF16 GEMV for every batch row and output row. No shared weight decode or multi-output register tile. | Add BF16-weight/F32-activation `MxN` AVX2/NEON kernels; on AVX512-BF16 use dot-product instructions; dispatch MOSS adaptor and dense BF16 prefill through them. |
| P0 | NVIDIA Q4 GEMM PTX: `gemm_q4sym` | One CUDA block computes one output scalar with a 256-thread shared-memory reduction. It reloads the same activation row and metadata independently for each output column and leaves little instruction-level reuse. | Replace with a 2-D output tile per block/warp. Decode packed weights vectorially, retain several column accumulators per thread, stage activation tiles once, and use warp reductions rather than nine block-wide barriers. |
| P0 | NVIDIA FP8/NVFP4 “GEMM” paths | Several runtime paths dequantise/transpose to F32 and call the simple SGEMM, or use one-output kernels. FP8 already includes same-input dual projection, batched QKV/O, SwiGLU-residual and DiT-island fusion helpers. | Build native tiled FP8/NVFP4 kernels with fused scale application; retain dequant-to-SGEMM as a shape-dependent fallback, improve existing FP8 fused dispatch, and extend comparable fusion to NVFP4. |

## Dense F32 and BF16 CPU kernels

| Kernel | Status against article | Further optimisation |
|---|---|---|
| amd64 `SgemmNT` | Good decode-oriented dot kernel; one output scalar at a time. | Add a true 3x4/4x4 AVX2 register tile for prefill. Existing blocked kernel only shares A across two output columns and does not tile activation rows. P1. |
| amd64 `sgemmNTTileFMA` | 64x64 cache blocking and 1x2 register tile; shape-aware dispatch exists for batches 2-256. | Expand to 2x4 or 3x4, use 12 accumulators, and retune K/N blocks per L1/L2. This should allow removing the current `batch<=256` guard after Whisper-shape proof. P1. |
| arm64 `SgemmNT` / blocked tile | SIMD exists, but current structure has fewer output accumulators than the article's ARM kernels and lacks architecture-specific FP16/BF16 dot paths. | Use 32 architectural vector registers for 4x4/4x8 tiles; add ARMv8.2 FP16/BF16 kernels and cache tuning. P1. |
| amd64/arm64 `SgemmNN` | Vectorises contiguous N and is suitable for pre-transposed weights, but uses fixed tiles and generic parallel column slicing. | Add M-row register tiling, packed static B panels, and caller-owned 2-D tiles. Benchmark BERT/Whisper attention and pre-transposed LLM projections separately. P1. |
| `SgemmNNParallelTo` | Threads contiguous N ranges, preserving arithmetic order. | Reuse a persistent worker pool; choose 2-D tile partition when M is large; avoid launching goroutines for marginal shapes. P2. |
| `GemvRows` / `GemvRowsParallel` | Correct decode path; one row per dot and repeated activation loads. | Add x4 output-row F32 dot kernels analogous to quant x4 helpers. This is useful for bandwidth/cache-resident small models, but decode remains weight-bandwidth bound. P2. |
| `GemvCols` / `GemmCols` | Scalar strided-column accumulation. | High-value if still used: route to `SgemmNN`, transpose/pack static weights once, or add contiguous-N outer-product kernels. P1. |
| `GemmRowsBF16Parallel` | Repeated BF16/F32 dot. | Native batched multi-row kernel as above. P0. |
| `GemvRowsBF16BF16Parallel` | Parallel one-output dots, capped at six workers. | Add x4 output rows and tune worker count from shape/cache rather than a fixed cap. P1. |
| `tensor.MatMul` / `MatMulTransposed` | Calls serial checked SGEMM regardless of shape. | Dispatch large tensors to parallel/tiled wrappers; `MatMulTransposed` should use the same shape-aware NT policy as model prefill. P1. |
| `models/whisper` attention SGEMMs | Per-head calls cause many small serial GEMMs. | Batch heads, pack K/V layouts for a head-group kernel, and fuse scale/softmax where practical. Multi-head call amortisation is more valuable than only changing the inner FMA loop. P1. |
| BERT, Hunyuan3D, Trellis2, speaker convolution SGEMMs | Direct serial `SgemmNNTo/NTTo` even for large matrices. | Route through a common shape-aware parallel dispatcher; add model-shape benchmarks before changing arithmetic order. P1/P2. |

## Quantised CPU kernels

| Family | Current design | Candidate optimisation |
|---|---|---|
| GPTQ Q4 symmetric/asymmetric | Non-RVV implementations are scalar package loops; runtime capability flags currently report no package-local assembly. GEMM often parallelises repeated GEMV over batch rows. | Add AVX2/NEON Mx4 kernels; share group-index/scale loads; prepack output-row groups. P0. |
| MLX Q4 | Row-wise group decode with parallel output rows for GEMV; batch repeats GEMV. | Native batched MxN microkernel and persistent packed panels. P0. |
| NVFP4 | Decode packed nibbles and block scales per row; batch repeats GEMV unless an accelerator exists. | Multi-row decode reuse and fused scale-vector kernels. P0. |
| FP8 E4M3 CPU | AVX2 dot uses a 256-entry F32 LUT. `Linear.BatchGemvTo` and dynamic-token variants already reuse each dequantised weight row across the batch, but remain row-oriented and internally serial. | Replace LUT gathers where conversion instructions/bit arithmetic win; add MxN register tiles and bounded output-tile threading. P1. |
| Q8/Q4 low-level dots | amd64 has single and x4 helpers. | Wire x4 helpers consistently through all row matrices; add arm64/NEON equivalents, which currently fall back to scalar for several helpers. P1. |
| Q5_0 and Q8_0 GGUF | Go scalar block loops, parallel rows. | AVX2/NEON/VNNI row kernels and x4 rows; reuse quantised activation. P0/P1. |
| Q2_K/Q3_K/Q4_K/Q6_K GGUF | Complex scalar unpacking, per-row scratch, RVV-specific alternatives in model code. | Centralise backend kernels, predecode scale metadata, x4 rows, and batched prompt kernels. Avoid duplicate model-local implementations. P0. |
| DiffusionGemma Q6 int8 dot | Dedicated amd64 assembly exists for one row. | Add x4 output rows and batch/active-expert scheduling; fuse dequant scale and expert weighting. P1. |
| GGUF expert matrices | `GemvExpertTo` and model-local expert code mix direct dots, dequant-once, and Sdotx4. | Group active experts by format/shape, process multiple tokens and output rows per packed tile, and retain dequantised hot experts in bounded cache. P1. |

## NVIDIA PTX kernels

| Kernel | Current design | Candidate optimisation |
|---|---|---|
| F32 `sgemm_nn` | Conventional 16x16 shared-memory tile, one output/thread, scalar K loop. | Use register-blocked outputs/thread (e.g. 4x4), vector loads, double-buffered shared tiles, transposed/padded shared B, and shape variants for skinny M. P1; mature libraries would be faster, but zero-CGo/runtime-PTX is a constraint. |
| Q4 `gemm_q4sym` | One block/output scalar and full shared reduction. | Complete redesign to tiled outputs. P0. |
| Q4_K/Q5_0/Q8_0 batch kernels | Mostly GEMV-style grid extended over batch; scatter variants add indirection. | Tile batch and output rows, share activation tiles, and provide pointer-table expert kernels that compute several rows/work items per block. P1. |
| MLX GEMM PTX | Separate dequant/correction work and general matrix kernel. | Fuse affine correction and nibble decode into tiled accumulators; specialise common group size 64/128 and prefill M ranges. P1. |
| FP8/NVFP4 | Conversion/dequant staging plus F32 SGEMM remains in several paths. FP8 already has same-input dual GEMM, batched QKV/O, SwiGLU-residual and DiT-island fusion helpers; NVFP4 has less fusion coverage. | Replace staging with native scaled tiled kernels where measured; improve dispatch and extend the existing fusion strategy, especially for NVFP4. P0. |
| LM-head GEMV kernels | One-token bandwidth-bound reductions. | Hierarchical fused argmax/top-k can avoid full logits download, but earlier local measurements found the current argmax slower. Retain GEMV by default; revisit only with fused projection+reduction. P2. |
| Whisper/attention matmuls | Dedicated attention kernels exist, avoiding generic score GEMM traffic. | Use warp-level QK tiles, online softmax, and V accumulation (FlashAttention-style) to avoid score materialisation. This is beyond the article but follows the same operand-reuse rule. P1. |

## SpacemiT, RVV and IME2

| Kernel | Status | Further optimisation |
|---|---|---|
| RVV `GemmI8Outer` M4N32 | Strong match to article: static B packing, multi-output outer product, output-tile ownership. | Add tails for M%4/N%32, cache-block K, persistent worker pool, and shape autotuning. P2. |
| RVV F16 outer/Outer32 | Already packed/output tiled. | Add architecture-tuned K blocking, fused bias/activation, and mixed BF16/F16 variants. P2. |
| RVV `GemmI8`/threaded basic | Basic kernels remain available beside outer kernels. | Ensure all eligible inference call sites select packed outer kernels; remove runtime repacking. P1. |
| RVV U8W4 | Quantised outer path exists. | Add multi-row activation quantisation, fused zero-point correction, tails and packed metadata. P1. |
| IME2 packed 4x4 vmadot | Good register tile and static packing; serial driver writes temporary 4x4 accumulators. | Use direct output-stride stores, larger composite tiles, pooled tile scheduling, and fuse quantise/dequant/bias/GELU epilogues. P1. |
| IME2 direct/simple kernels | Useful fallbacks but less reuse and packing. | Restrict to tiny/one-shot shapes; route static inference weights to packed/pool path. P2. |
| AIPool Q8x32 X100-pack | Already persistent pool and packed accelerator work. | Co-schedule paired QKV/gate-up products sharing activations and fuse epilogues; benchmark queue overhead for small M. P2. |
| AICPU Q4_K/Q6_K/Q8x32 native pooled and mixed-batch paths | `backends/spacemit/aicpu` already includes Q4_K pooled/M4 and mixed-batch, Q6_K native repack, and Q8x32 native pooled kernels. | Consolidate format-specific scheduling, reuse packed activations across paired products, add tail/shape dispatch, and compare native packed formats against IME2 conversion overhead. P1/P2. |
| Vulkan `GemvF32` | Board wrapper exposes GEMV; implementation maturity is limited. | Add batched GEMM and persistent device weights before microkernel tuning. P2/P3. |

## Model-local bypasses and orchestration opportunities

These are not all new arithmetic kernels, but they prevent existing kernels from seeing enough work to reuse data:

- `model/diffusiongemma/cpu_dispatcher.go` contains local fallback `dot` loops, although its main GGUF expert path already groups `expertUsers` and includes Q4_K/Q5_0/Q8_0 batched row kernels. Optimise the remaining format/tail fallbacks rather than rebuilding its existing batching.
- `model/gguf_llama.go` and `loader/gguf/expert_matrix.go` retain scalar F32 row dots. Consolidate them on `simd.GemvRowsParallel`/x4 variants. `loader/gguf/quant_project.go` already has quantised batch paths; only its scalar/tail fallbacks need similar consolidation.
- `model/internal/ops.GemvNT` should use the checked backend facade and parallel row dispatch for large outputs.
- `models/whisper/decoder.go` already uses `Sdotx4` for attention scores; add head-group and query-group batching rather than increasing dot width alone.
- `model/mosstranscribe/adaptor.go` is a concrete BF16 batched-GEMM target.
- `model/cpu_prefill.go` correctly distinguishes prefill from decode for dense F32, but quantised `projBatch` routes still depend on repeated-GEMV implementations.
- MoE paths outside the existing DiffusionGemma `expertUsers` batching should group tokens by active expert and projection type. Per-token expert GEMV leaves cross-row reuse unavailable when several tokens select the same expert.
- Convolution implementations lowered to `SgemmNNTo` should pack im2col/weights once per stable shape and use parallel GEMM; direct convolution remains preferable when im2col traffic dominates.

## Kernels that should not be changed first

- Single-token dense decode GEMV: it is primarily weight-bandwidth bound. Wider output-row kernels may help cache-resident small models but will not deliver article-like prompt gains on large weights.
- Fused GPU LM-head argmax: existing local evidence says the current GPU reduction loses to logits download. A redesign must fuse projection and reduction before defaulting it.
- RVV M4N32 and IME2 4x4 packed microkernels: these already embody the article's main ideas; orchestration, tails and epilogues are higher-value than replacing them.
- Large Whisper F32 batches on the current 1x2 blocked NT kernel: the documented local MOSS measurements in `simd-matmul.md` showed an end-to-end regression, hence the `batch<=256` guard. Widen the microkernel before widening dispatch.

## Recommended implementation order

1. Quantised CPU prompt kernels: MLX Q4, GPTQ Q4, GGUF Q4_K/Q6_K, NVFP4.
2. BF16 MxN kernel for MOSS adaptor and BF16 checkpoints.
3. Dense AVX2/NEON 3x4 or 4x4 NT microkernel, then retune dispatch across Whisper-sized M.
4. NVIDIA Q4 tiled GEMM redesign.
5. Native NVIDIA FP8/NVFP4 tiled kernels with fused scales and paired projections.
6. Common shape-aware SGEMM dispatcher for tensor, BERT, Whisper, speaker, Hunyuan3D and Trellis2 call sites.
7. MoE token/expert batching and x4 quant row kernels.
8. RVV/IME2 tails, epilogues and persistent scheduling improvements.

Each item needs three gates: kernel numerical parity, a stable shape matrix covering decode/prefill/tails, and at least one end-to-end model fixture. Microbenchmarks alone are insufficient because packing, goroutine creation, activation quantisation and cache pressure often reverse isolated wins.
