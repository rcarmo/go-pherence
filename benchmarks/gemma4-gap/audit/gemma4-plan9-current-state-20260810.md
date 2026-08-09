# Gemma 4 Plan 9 assembly checkpoint — 2026-08-10

## Pre-commit repository state

- Worktree: `/workspace/projects/go-pherence-gemma-plan9`
- Branch: `perf/gemma4-all-plan9`
- Base before this batch: `d08ce322292847a7978dcace1cc73a50d0bf8450`
- Dirty `/workspace/projects/go-pherence-plan9`: untouched
- `git diff --check`: clean

This section records the isolated worktree state before the validated batch was committed; push and the final post-change profile remain separate decisions.

## Implemented and retained

### Fused Q4_0×Q8_0 projection

`loader/gguf/llamaq4plan9` now contains independent Plan 9 assembly kernels and no cgo import:

- compiler-translated 8×4 and 8×16 correctness baselines;
- the exact stage-local 8×8 kernel with a 584-byte native subtraction;
- direct strided stores for complete eight-row groups;
- a paired direct 8×16 entry point which inlines two staged pairs under one native entry, uses an 824-byte total manual frame, and reuses the accepted staged arithmetic;
- row and token tails through the exact 8×8/8×4 paths;
- AVX2/AVX-VNNI/FMA dispatch checks and retained unsupported-CPU/cgo/portable fallbacks.

The paired kernel does not add another weight representation. Its only new permanent data is a 256-byte vector-constant table. Production weight ownership and activation preparation remain unchanged.

The projection scheduler is restored to dynamic `workers*4` chunking. `projectQ4_0LlamaExperimentalBackend` exists to compare Plan 9 and retained cgo under identical scheduling; normal production dispatch selects Plan 9 when available.

### Exact Q8 preparation

The already-promoted exact AVX2 Plan 9 block quantizer remains in:

- `loader/gguf/q8_quant_amd64.go`
- `loader/gguf/q8_quant_amd64.s`

It retains scalar/non-amd64 fallbacks and the exact near-half, NaN, infinity, saturation, scale, and direct-to-SoA behavior previously accepted.

### Exact FP16-table GELU×up

`internal/ggmlfp16` now has an AVX2/F16C Plan 9 kernel which:

- reproduces the repository's nonstandard FP32→FP16 rounding algorithm with packed integer operations;
- gathers the immutable FP16 GELU table without creating a duplicate table representation;
- preserves `x <= -10 → 0`, `x >= 10 → x`, NaN/infinity behavior, FP16 widening, multiply order, in-place aliasing, and scalar tails;
- checks AVX2 through `x/sys/cpu` and F16C through a small CPUID Plan 9 helper;
- retains scalar and non-amd64 fallbacks.

The GELU table has one extra `uint16` sentinel solely so the final two-byte-scaled dword gather at index `0xffff` remains inside its allocation.

`model.ggmlGELUMulInPlace` now uses this exact shared implementation.

### amd64 partial RoPE

`backends/simd/runtime` now has an AVX2 Plan 9 partial-RoPE kernel which:

- handles full and partial rotations;
- preserves head-major layout and untouched dimensions;
- uses separate multiply/add/subtract operations (no FMA contraction), retaining exact scalar FP32 results;
- handles vector and scalar pair tails in assembly;
- keeps malformed-input behavior and scalar/non-amd64/RISC-V fallbacks.

Runtime capabilities now report RoPE assembly on amd64 only when the normal AVX2 capability is available.

### Preserved generator sources

The non-production C intrinsic sources used to generate/audit the assembly are retained under:

`benchmarks/gemma4-gap/audit/plan9-kernel-sources-20260810/`

They include the staged/direct Q4 sources, paired Q4 source, exact GELU source, and partial-RoPE source. The production build does not compile these C files.

## Correctness evidence

Passed:

- fused tile parity against retained llama topology for every block count `1..80`, now including the paired 8×16 kernel;
- projection parity across row/token tails and dynamic scheduling;
- exact exhaustive GELU checks over all 65,536 FP16 inputs, threshold-adjacent values, random FP32 bit patterns, NaN/infinities, aliases, and lengths `0..23`;
- exact RoPE checks across heads `1..9`, even head dimensions `2..130`, full/partial/vector/scalar tails, and untouched dimensions;
- explicit unsupported-CPU fallback tests for GELU and RoPE;
- focused package suites with cgo and `CGO_ENABLED=0`;
- arm64 and riscv64 test-binary cross-compilation for the changed GELU and SIMD runtime packages;
- complete `go test ./... -count=1` with normal cgo configuration;
- focused race suites for runtime SIMD, GELU, Q4 Plan 9, GGUF, and model packages;
- `go vet` for the changed GELU, Q4 Plan 9, GGUF, and model packages;
- finite real-model frozen request;
- real one-step decode/checkpoint/restore/logit test.

Focused command set:

```sh
go test ./backends/simd/runtime ./internal/ggmlfp16 \
  ./loader/gguf/llamaq4plan9 ./loader/gguf ./model -count=1
CGO_ENABLED=0 go test ./backends/simd/runtime ./internal/ggmlfp16 \
  ./loader/gguf/llamaq4plan9 ./loader/gguf ./model -count=1
```

`CGO_ENABLED=0 go test ./...` still fails outside this scope in the pre-existing SpaceMIT AICPU packages because `ime2.Vmadot*` symbols are unavailable. The changed focused packages pass with cgo disabled. Whole-package `go vet` for `backends/simd/runtime` also reaches a pre-existing `q8dot_amd64.s` ABI-offset diagnostic; vet passes for the other changed packages, and the SIMD runtime race/test suites pass.

## Performance evidence

All timings used `taskset -c 0-5` and `GOMAXPROCS=6`.

### Q4 projection

The first grouped benchmark was distorted by sustained-load ordering and made Plan 9 appear about 2.1% slower. An order-alternating benchmark then measured both backends inside every benchmark iteration with identical packing, scheduler, and load conditions.

Seven three-second samples:

| Backend | Median |
| --- | ---: |
| Plan 9 paired 8×16 | **116,668 ns/op** |
| Retained cgo 8×16 | 121,676 ns/op |
| Plan 9 / retained median ratio | **0.9577** |

Plan 9 won all seven alternating samples, by approximately **4.2%** at the median. The paired standalone tile median was 3,304 ns versus 3,400 ns for two staged 8×8 calls, a 2.8% mechanism win.

### Exact GELU×up

At `124 × 10,240` values:

| Implementation | Median |
| --- | ---: |
| Plan 9 AVX2/F16C | **1.880 ms** |
| Retained scalar lookup loop | 18.576 ms |

The exact assembly path is approximately **9.9× faster**, with no allocation.

### Partial RoPE

At 16 heads, head dimension 256, rotation half 128:

| Implementation | Median |
| --- | ---: |
| Plan 9 AVX2 | **30.701 µs** |
| Retained scalar Go | 62.670 µs |

The exact assembly path is approximately **2.0× faster**, with no allocation.

## Frozen 124+48 result

Three consecutive pinned candidate runs reproduced the accepted fused token trajectory exactly:

```text
[174982 1123 7276 241685 11344 3111 1200 9222 1363 3136 9683 9683
 19677 2835 106289 19715 9683 6825 246293 68252 68252 110537 68252
 56205 30244 9222 7276 35106 7276 237096 238914 66916 9222 9222
 9222 111417 19715 208121 9222 237072 241546 9222 237062 236840
 237335 384 9222 9222]
```

| Metric | Three-run values | Median |
| --- | --- | ---: |
| Prompt compute tok/s | 89.993, 89.506, 90.700 | **89.993** |
| Decode-evaluation tok/s | 9.484, 9.370, 9.355 | **9.370** |

The frozen prompt median clears the explicit **89.405 tok/s** target by about 0.66% and reaches 98.64% of the 91.229561 tok/s llama.cpp oracle. A prior standalone run reached 91.357 prompt tok/s; it is not included in the three-run median.

Raw evidence: `gemma4-plan9-124x48-20260810.log`.

The checkpoint/restore gate also passed separately:

```text
TestGemma4DecodeSessionUpdatedRealGGUFOneStep — PASS (9.75s)
```

## Remaining work after the validated code commit

1. **Review generator provenance before any later assembly regeneration**
   - decide whether to add a small reproducible generator script rather than relying on the preserved sources and documented pipeline;
   - retain the compiler baselines as correctness references, not production dispatch.
2. **Take one post-change prefill CPU profile**
   - confirm the 9.9× GELU improvement removed the prior 7% hotspot;
   - decide from measured samples whether the remaining Go softmax (previously 0.53%) or argmax warrants assembly. Do not add them merely for nominal coverage if still immaterial.
3. **Optional final throughput confirmation**
   - one additional clean three-run frozen set or an order-balanced clean-main/current pair if the machine load is stable;
   - retain the accepted 89.993 tok/s median unless contrary clean evidence appears.
4. **Finalize audit and ship**
   - update the kernel inventory statuses for Q4, GELU, and RoPE;
   - record any consciously retained scalar orchestration/transcendental paths;
   - push `perf/gemma4-all-plan9` only when requested and verify the remote branch.

## Stop point

This is a safe end-of-day checkpoint: all new code and generator sources are preserved in the isolated branch, tests and real-model gates pass, the target is met, no push has occurred, and the unrelated dirty main worktree remains untouched.
