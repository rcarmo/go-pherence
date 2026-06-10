# Ideogram 4 on SpaceMIT K3

This document tracks the K3-specific implementation plan for native Ideogram 4 inference on the SpaceMIT K3/MilkV Jupiter 2 class of systems.

## Target

- CPU: SpaceMIT K3/X60, RVV 1.0.
- AI cores: A100 cluster, cores 8–15, IME2 integer matrix extension.
- Scratch: `/dev/tcm`, 8 × 384 KB = 3 MB.
- Thread placement: `backends/spacemit/k3engine/aipool` must register/pin workers through `/proc/set_ai_thread` before scheduling IME2 work on cores 8–15.
- Memory policy target: 24 GB system RAM. Prefer resident component/activation buffers and bulk staged transfers over tiny repeated allocations.

## Existing K3 kernel assets

Relevant packages:

```text
backends/spacemit/ime2       IME2 vmadot int8/i4/q4 kernels and WorkerPool
backends/spacemit/rvv        RVV kernels: int8, fp16, quantization, copy
backends/spacemit/tcm        TCM mapping
backends/spacemit/k3engine   A100 worker-pool based transformer engine
backends/k3                  higher-level K3 backend dispatch
```

Important existing primitives:

```text
ime2.K3I8I8M1/M4/M1Groups
ime2.K3I8I4* / Q4K kernels
rvv.GemmF16Outer32 / DotF16 / F32ToF16RVV
rvv.MatMulIntegerDequant / QuantizeWeightsSym / QuantizeDynamicU8
inference.RMSNorm / QuantizeF32ToINT8 / MatVecQ4K*
aipool.NewAIWorkerPool / RegisterAIThread
```

Assembly macro/reference files:

```text
backends/spacemit/ime2/k3_isa.h
  Canonical hand-encoded instruction macros for K3 custom IME2 ops and RVV/Zvfh
  helper instructions. Use this instead of scattering raw WORD encodings.

backends/spacemit/ime2/*_riscv64.s
  Existing IME2 vmadot, i8×i8, i8×i4, Q4_K, grouped GEMM, and argmax kernels.

backends/spacemit/rvv/*_riscv64.s
  Existing RVV copy, q8 quantization, fp16 conversion, fp16 dot/GEMM, and
  w4/i8 kernels.
```

For new Ideogram K3 kernels, follow the existing style:

```asm
#include "textflag.h"
#include "../ime2/k3_isa.h"
```

and prefer named macros like `VMADOT_SS`, `VMADOT_SU`, `K3_VSETVLI_*`,
`K3_VLE*`, `K3_VSE*`, and `K3_VFWMACC_*` over raw hex encodings.

## Current build status

The generic native Ideogram CLI and model tests cross-compile for riscv64 today.
Use the consolidated check target from an x86 development host:

```bash
make ideogram4-k3-check
```

It runs native Ideogram/NVIDIA fallback tests, cross-compiles riscv64 test
binaries for the Ideogram model/CLI packages and relevant K3 packages (including
`k3engine` and `aipool`), and builds `bin/ideogram4gen-k3` plus
`bin/ideogram4vaeprobe-k3`.
Equivalent raw commands:

```bash
mkdir -p .gotmp /workspace/tmp/ideogram4
GOTMPDIR=$PWD/.gotmp CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 \
  go build -o /workspace/tmp/ideogram4/ideogram4gen-k3 ./cmd/image/ideogram4gen
GOTMPDIR=$PWD/.gotmp CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 \
  go build -o /workspace/tmp/ideogram4/ideogram4vaeprobe-k3 ./cmd/image/ideogram4vaeprobe
GOTMPDIR=$PWD/.gotmp CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 \
  go test -c -o /workspace/tmp/ideogram4/ideogram4-model-k3.test ./model/ideogram4
```

This is **not yet full K3 SIMD coverage**. It is the first targetable binary for hardware smoke tests while K3 kernels are wired in.

## Ideogram kernel coverage matrix

| Area | Current generic path | K3 SIMD/IME target | Status |
|---|---|---|---|
| FP8 E4M3 linear GEMV/GEMM | scalar/amd64/NVIDIA-specific paths; K3 gate routes to RVV f16 GEMM bridge when `GO_PHERENCE_IDEOGRAM4_K3=1` on riscv64 | K3 packed FP8→int8 or FP8→f16 kernels using IME2/RVV | partial: RVV f16 bridge with resident decoded fp16 and N32-packed fp16 weight cache, not final IME2 |
| FP8 E4M3 decode | LUT scalar; K3 bridge decodes row-scaled FP8 to resident fp16 weight rows before RVV GEMM | RVV byte→f16/int8 packing, row-scale fused | partial |
| Qwen text encoder linears | `FP8Linear.ApplyBatch` | K3 FP8 batch linear, A100 worker-pool scheduling | missing |
| DiT QKV/O/W1/W2/W3 linears | `FP8Linear.ApplyBatch` / GPU on NVIDIA | K3 full-layer packed/resident linears | missing |
| RMSNorm rows | Go scalar / NVIDIA rows; K3-gated riscv64 seam covers Qwen/DiT weighted RMSNorm call sites | RVV row RMSNorm over f32/f16 | partial seam: scalar body pending RVV assembly |
| LayerNorm final | Go scalar / NVIDIA rows | RVV row LayerNorm | missing |
| RoPE/MRoPE | Go scalar / NVIDIA kernels | RVV in-place rotation | missing |
| Attention score/value | Go scalar / NVIDIA full attention | RVV/f16 tiled attention; future tiled/streaming for high res | missing |
| SiLU / SiLU*Mul / gated residual | Go scalar / NVIDIA vector kernels | RVV vector kernels | missing |
| CFG + scheduler update | Go scalar / NVIDIA vector kernel; K3-gated riscv64 seam exists | RVV vector kernel | partial seam: scalar body pending RVV assembly |
| VAE Conv2D | SIMD im2col/GemmRows / NVIDIA direct conv; K3 gate can route im2col GEMM through RVV fp16 bridge | RVV/f16 or int8 im2col/conv; potentially reuse f16 GEMM | partial: RVV f16 bridge |
| VAE GroupNorm/SiLU/Upsample/RGB | Go scalar / NVIDIA kernels | RVV vector kernels | missing |
| VAE spatial attention | Go/NVIDIA full attention | tiled/streaming RVV/f16 attention | missing |
| Residency/activation policy | NVIDIA hidden-resident/full-layer/windowed | K3 24GB resident component buffers + A100 worker-pool execution | design needed |

## Implementation direction

1. Keep the Ideogram model code backend-neutral. Add K3-specific acceleration behind package-level helpers instead of forking `model/ideogram4`.
2. Start with FP8 linears, because Qwen and DiT throughput depends on them.
3. For K3, do not attempt NVIDIA-style GPU residency. Instead, use 24 GB RAM for resident decoded/packed weights and activation work buffers, and use `aipool`/TCM for compute kernels. The first FP8 bridge already caches decoded fp16 weights plus the RVV N32-packed fp16 layout per `FP8Linear` to avoid re-decoding/repacking large projections on every call.
4. Prefer IME2 int8 for large linears if accuracy is acceptable after FP8 row-scale conversion; prefer RVV f16/f32 for norm/attention/vector kernels.
5. Maintain scalar fallbacks and cross-build tests until real K3 hardware validation is available.

## First handoff command

After implementing enough K3 kernels to make the path useful, the intended hardware smoke will be:

```bash
make ideogram4gen-k3
./bin/ideogram4gen-k3 \
  -k3 -k3-threads 8 -k3-prewarm \
  -model /path/to/ideogram4-model \
  -prompt "$(cat prompts/ideogram4/cat.json)" \
  -width 256 -height 256 -steps 4 \
  -guidance 7.0 -mu 0.0 -std 1.75 \
  -seed 2026060803 \
  -timing
```

A smaller VAE-only smoke is also available:

```bash
./bin/ideogram4vaeprobe-k3 \
  -k3 -k3-threads 8 -k3-prewarm \
  -model /path/to/ideogram4-model \
  -width 256 -height 256
```

K3-specific runtime environment should include:

```text
GO_PHERENCE_IDEOGRAM4_K3=1          # enables current riscv64 RVV f16 FP8-linear bridge
GO_PHERENCE_IDEOGRAM4_K3_THREADS=8  # optional RVV thread count for the bridge
GO_PHERENCE_IDEOGRAM4_K3_PREWARM=1  # pre-decode resident fp16 linears for 24GB profile
IME2_TCM_ACT=1                      # for future IME2/TCM-backed kernels
```

The same K3 switches can be set through CLI flags on `ideogram4gen`:

```text
-k3
-k3-threads 8
-k3-prewarm
```

and hardware logs should confirm A100 worker placement on cores 8–15.
