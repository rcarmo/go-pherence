# Gemma 4 CPU kernel inventory and frozen contracts

Date: 2026-08-10  
Branch: `perf/gemma4-all-plan9`  
Model: `gemma-4-E4B_q4_0-it.gguf`  
Scope: CPU prompt/decode path exercised by the frozen `124+48` request; GPU and DiffusionGemma are out of scope.

## Evidence boundary

The dynamic source is `fused-8x16-prefill-pprof-top-20260809.txt` (7.56 CPU-seconds). It accounts for 98.28% of samples. Static call-path inspection covers `model/cpu_prefill.go`, `model/forward_layer.go`, `model/attention.go`, `model/inference_helpers.go`, `loader/gguf`, and `backends/simd/runtime` so short kernels omitted by sampling are not silently excluded.

“Plan 9” below means a Go ABI wrapper plus Go assembler, not C generated through cgo. Go scheduling/orchestration and transcendental/table semantics remain Go unless a replacement can preserve the frozen contract.

## Inventory

| Gemma operation | Production implementation | Class | Dynamic evidence / disposition |
|---|---|---|---|
| Q4_0×Q8_0 fused projection, 8 rows × 16 tokens | `loader/gguf/llamaq4/kernel_amd64.c`, reached through `ProjectQ4_0x8Q8_0x4RowsVNNI` | **C/cgo, must replace** | `runtime.cgocall` 81.08%; dominant target. Compiler-translated Plan 9 8×16 baseline is bit-identical but does not beat retained C because its ~1312-byte aligned frame still spills heavily. |
| Q4_0×Q8_0 tail projection, 8×4 | same C TU | **C/cgo, must replace** | Direct compiler-translated Plan 9 baseline is bit-identical for blocks 1…80 and faster as a standalone tile; integrate only with complete row/token tails. |
| Q8_0×4 activation preparation | `loader/gguf.quantizeQ8_0x4To` Go traversal plus scalar conversion/rounding | **Go, must replace hot block work** | 6.88% direct plus `half.F32ToF16`, `math.Round`; existing exact AVX2 block quantizer work is the semantic oracle. Scheduling may remain Go. |
| Q6_K×Q8_K projection / LM-head support | `dotQ6KQ8KGemvVNNIAsm`, `qdot_q6_coeff_amd64.s` | **Plan 9 complete** | 1.19%; retain dispatch and scalar fallback. |
| F32/BF16 GEMV and GEMM | `sdotAsm`, `dotRowsx4Asm`, `gebpMicroKernel`, `SgemmNN/NT`, blocked/gather kernels | **Plan 9 complete for arithmetic** | Used by dense/non-quantized and verifier paths; Go row/worker scheduling remains orchestration. |
| F32 vector add/mul/scale/scale-add | `vec_amd64.s`, `simd_amd64.s` | **Plan 9 complete** | Used for residuals, scaling, and attention value accumulation. |
| RMSNorm / BF16 RMSNorm / no-scale norm | `vec_amd64.s` | **Plan 9 complete** | Static Gemma graph coverage; preserve F32 accumulation and in-place behavior. |
| F32↔BF16 conversion | `vec_amd64.s` | **Plan 9 complete** | `ToBF16`, widen, narrow are already assembly-dispatched. |
| Q/K/V RoPE partial rotation | scalar Go on amd64 (`hasRoPEAsm=false`; only RISC-V has accelerated head kernel) | **Go, missing** | Static Gemma graph coverage. Needs an amd64 Plan 9 head kernel while retaining scalar malformed/tail behavior. |
| Attention score dot products | `sdotAsm` | **Plan 9 arithmetic complete** | 0.66% `sdotAsm`; Go GQA/head scheduling is orchestration. |
| Attention softmax | Go max/`math.Exp`/sum plus Plan 9 `VecScale` | **Go transcendental, missing** | 0.53% inclusive in sampled prompt. Any assembly replacement must preserve finite/NaN/Inf behavior and reduction contract; not ahead of projection. |
| Attention value reduction | repeated `VecScaleAdd` | **Plan 9 arithmetic complete** | Go token loop remains orchestration. |
| Gemma exact FP16-table GELU×up | `model.ggmlGELUMulInPlace` and `internal/ggmlfp16.GELUFP16Lookup` | **Go, missing hot loop** | 7.28% inclusive / 7.01% lookup. Assembly may accelerate indexing/multiply only; the frozen FP16 table and input rounding are immutable. |
| SiLU/tanh GELU fallback variants | scalar Go non-linearity plus Plan 9 multiply | **Go transcendental** | Not the dominant E4B exact-GELU path. Keep fallback; do not substitute approximation into the exact table path. |
| Embedding gather/copy | Go slice copy/conversion, with Plan 9 BF16 widening where applicable | **Go memory orchestration** | No standalone hotspot. Not a numerical kernel worth an ABI addition unless a later profile makes it material. |
| Final logits projection | Q4/Q6 quantized projection or Plan 9 GEMV according to tensor type | **Covered by quantized projection / Plan 9** | Preserve output-row ordering and suppression/soft-cap sequence. |
| Logit soft-cap / suppress tokens | scalar Go `math.Tanh` and indexed stores | **Go post-processing** | Not sampled materially; preserve exact ordering. |
| Argmax | scalar Go `ArgmaxLogits` | **Go, missing but non-material** | Preserve first-maximum tie behavior and empty-input error. |
| Checkpoint/KV restore | Go state/memory orchestration | **Not an arithmetic kernel** | Correctness gate, not an assembly target. |

## Frozen ABI, layout, and fallback contracts

### Common amd64 dispatch

* AVX2 kernels execute only when their declared feature set is present. VNNI fused Q4 additionally requires `AVX-VNNI`; scale accumulation requiring FMA keeps the FMA check.
* Unsupported CPUs take the retained portable implementation or return the existing unsupported error. No illegal instruction may be reachable through an unchecked symbol.
* `CGO_ENABLED=0` must build and run with portable behavior. The Plan 9 package must not import or link cgo.
* Assembly entry points are internal `//go:noescape` functions. Public wrappers validate sizes before obtaining slice data pointers.

### Fused Q4 projection

* Q4 panel: one 8-row group is `blocks*144` bytes; each block starts with eight FP16 row scales followed by four 32-byte packed row-pair payloads. The fused llama topology consumes the production `0x88` transformed nibble representation.
* Q8 panel: one four-token panel is `blocks*136` bytes; each block contains four FP16 token scales followed by four 32-byte token payloads. Four panels form a 16-token supertile and are separated by `blocks*136` bytes.
* Tile output is token-major `[tokens][8]` F32. Whole projection output is token-major `[tokens][rows]` F32.
* Full 8×16 tiles may use llama-style reduction and need only match the retained portable llama-topology reference, not legacy dot ordering. 8×4 tails must use the same production packed-weight interpretation.
* Row tails write only logical rows; token tails write only logical tokens. No padded lane may escape into the logical output.
* `blocks<=0`, short buffers, unsupported features, and nil/empty inputs retain current errors. Output/input overlap is not promised unless the current public wrapper explicitly supports it.
* Six-core scheduling assigns disjoint row groups and must not nest a second worker pool. Caller participation is allowed; output must be deterministic for a fixed topology.

### Q8 preparation

* Exact block selection, absolute-max tie behavior, zero block, FP16 scale bits, inverse-scale reconstruction, round-to-nearest behavior, int8 saturation, and four-token SoA byte layout are frozen by the retained quantizer tests.
* Preparation remains transient; no permanent duplicate activation or weight representation may be introduced.

### RoPE

* Layout is head-major contiguous F32; for every head, pair `i` is `(x[i], x[i+rotHalf])`. Unrotated dimensions remain byte-for-byte unchanged.
* Frequency layout and position indexing remain those produced by `BuildRoPEFreqs*`.
* Invalid shapes continue to fail through `ApplyRoPEPartialTo`; scalar tails and unsupported CPUs use the Go reference.

### Exact GELU

* Inputs in `[-10,10]` are converted to FP16 with the existing repository conversion, index the immutable ggml FP16 GELU table, widen the FP16 table value, and multiply by `up` in the current order.
* Out-of-range, NaN, infinities, aliases (`dst==gate`), and tails retain reference behavior.

### Softmax and argmax

* Softmax remains numerically stable (`max`, exponentiate, serial F32 sum, validate sum, reciprocal scale) unless a replacement proves the accepted functional contract and finite model run.
* Argmax returns the first maximum; NaN behavior and empty-input error are unchanged.

## Priority after inventory

1. Replace the fused Q4 C/cgo path with a spill-reduced Plan 9 kernel and complete projection integration.
2. Promote the already-proven exact AVX2 Q8 block quantizer into the winning projection batch.
3. Accelerate exact GELU table lookup/multiply.
4. Add amd64 partial-RoPE assembly.
5. Consider softmax/argmax only after a new whole-request profile shows material budget.

The compiler-translated 8×4/8×16 package is a correctness and code-generation baseline, not the production candidate. Its fused frame and stack traffic explain why language-boundary removal alone cannot close the throughput gap.
