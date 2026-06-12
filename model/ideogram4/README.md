# model/ideogram4

Ideogram v4 image generation: a DiT (diffusion transformer) with a Qwen3-VL
conditioner and an AutoencoderKLFlux2 VAE decoder. FP8-weight native path plus
GPU (CUDA) variants.

## Architecture

| Area | Files |
|---|---|
| Pipeline / runtime | `native_pipeline.go`, `runtime.go`, `cfg.go`, `config.go`, `denoise_loop.go` |
| DiT transformer | `dit_block.go`, `dit_forward.go`, `latents.go`, `latent_norm.go` |
| Conditioning | `conditioning.go`, `qwen_vl_conditioner.go` |
| Scheduler | `scheduler.go`, `scheduler_runtime.go` |
| FP8 weights | `fp8_layout.go`, `fp8_linear.go`, `fp8_load.go` |
| VAE decoder | `vae_decoder.go`, `vae_ops.go` |
| GPU / NVIDIA | `gpu_*.go` (attention, mlp, mrope, norm, VAE conv/groupnorm, residency) |
| K3 / SpacemiT | `k3_*.go`, `gpu_k3_policy.go` |
| Inventory / inspect | `tensor_inventory.go` |

## K3 / SpacemiT native path

When `-k3` (or `GO_PHERENCE_IDEOGRAM4_K3=1`) is set, the pipeline runs entirely
on the K3 SoC using X100 general cores and A100 AI cores. All NVIDIA/GPU paths
are hard-disabled via `gpu_k3_policy.go` — there is no escape hatch because K3
has no NVIDIA hardware.

### K3 files

| File | Purpose |
|---|---|
| `gpu_k3_policy.go` | `gpuDisabledByK3()` — hard-disables all Ideogram GPU predicates under K3 |
| `k3_fp8_riscv64.go` | A100 row-scale Q8 FP8 linear dispatch, worker pool, FP8→Q80 packer |
| `k3_fp8_other.go` | Non-riscv64 stubs |
| `k3_fp8_a100_riscv64_test.go` | Correctness/benchmark tests for A100 Q8 path |
| `k3_mlp_riscv64.go` | Fused A100 MLP W1/W3/W2, RVV SiLU×Mul |
| `k3_mlp_other.go` | Non-riscv64 stubs |
| `k3_attention_riscv64.go` | Flash Attention with online softmax, parallel row workers, SIMD Sdot/Saxpy |
| `k3_attention_other.go` | Non-riscv64 stubs |
| `k3_gemm_riscv64.go` | A100 Q8 GEMM for VAE Conv2D with weight caching |
| `k3_gemm_other.go` | Non-riscv64 stubs |
| `k3_cfg_riscv64.go` | RVV vector CFG/FlowMatch seam |
| `k3_norm_riscv64.go` | RVV RMSNorm seam |
| `k3_rope_riscv64.go` | RVV RoPE/MRoPE seam |
| `k3_vae_riscv64.go` | Parallel GroupNorm, upsample, RGB conversion |
| `k3_vae_attention_riscv64.go` | VAE spatial attention seam |
| `k3_prewarm.go` | Selective DiT/Qwen prewarm with raw-weight release |
| `k3_prewarm_riscv64.go` | riscv64 prewarm dispatch |

### How A100 cores are used

All 8 A100 cores (8–15) are registered via `/proc/set_ai_thread` and
`sched_setaffinity`. The worker pool (`backends/spacemit/k3engine/aipool`) pins
goroutines to A100 cores and dispatches Q8×Q8 IME2 GEMM kernels (`K3I8I8M4/M1`).

FP8 weights are converted to row-scale Q80x32 A100 tile layout using a fused
single-pass parallel packer that decodes each E4M3 byte once. Packed weights are
cached per tensor pointer and optionally released (raw FP8 bytes freed) after
prewarm to halve peak memory.

Activation packing runs on X100 goroutines before A100 dispatch. The activation
A-block buffers are pooled via `sync.Pool` to reduce GC pressure.

### How X100 cores are used

X100 cores 0–7 run:
- Activation Q8 M4 packing (parallel across row groups)
- Im2col for VAE Conv2D (parallel across rows)
- GroupNorm (parallel across groups)
- Flash Attention row workers (parallel across head×token jobs)
- RVV polynomial SiLU×Mul
- SIMD Sdot for Q·K scoring
- SIMD Saxpy for V accumulation
- FastExp (Schraudolph) for softmax
- RoPE/MRoPE
- RMSNorm
- CFG/FlowMatch vector updates
- Output transpose + bias (parallel across channels)

### Performance (Milk-V Jupiter 2 / K3 SoC, 31 GB LPDDR)

| Resolution | Steps | Total generation | Per-step cached |
|---|---:|---:|---:|
| 64×64 | 1 (prewarmed) | 12.7s | 4.2s |
| 128×128 | 1 (prewarmed) | 27.4s | 13.6s |
| 256×256 | 1 | 3m24s | — |
| 256×256 | 4 | 6m42s | ~64s |
| 512×512 | 1 | 15m22s | ~9m27s |

### Env flags

```sh
GO_PHERENCE_IDEOGRAM4_K3=1                  # enable K3 native path
GO_PHERENCE_IDEOGRAM4_K3_A100_Q8=1          # A100 row-scale Q8 for FP8 linears
GO_PHERENCE_IDEOGRAM4_K3_A100_MLP=1         # fused A100 MLP W1/W3/W2
GO_PHERENCE_IDEOGRAM4_K3_A100_WORKERS=8     # A100 worker count (default 8)
GO_PHERENCE_IDEOGRAM4_K3_PREWARM_QWEN=1     # prewarm Qwen text conditioner
GO_PHERENCE_IDEOGRAM4_K3_FAST_SILU=lut      # opt-in LUT SiLU (visible quality change)
GO_PHERENCE_IDEOGRAM4_TIMING=1              # enable per-layer/sublayer timing
IME2_ACT_PACK_WORKERS=8                     # activation packing parallelism
```

CLI flags: `-k3`, `-k3-prewarm`, `-k3-threads N`, `-timing`.

### Memory constraints

- 31 GB total on the Milk-V board.
- Each transformer branch's Q80 caches ≈ 11.3 GB.
- Both branches + Qwen + VAE + runtime > 31 GB → only cond branch can be prewarmed.
- Uncond branch builds Q80 caches lazily on first use (fast parallel packer).
- Raw FP8 weights are released after prewarm via `ReleaseRawWeights()`.
- Inter-step GC prevents OOM during multi-step generation at 256×256.
- `releaseDenoiseBeforeVAE` drops tokenizer/text/DiT before VAE decode.

## Reusable backend code

These packages contain kernel work done for Ideogram4 but usable by any model:

### `backends/spacemit/rvv`

| File | Function | Notes |
|---|---|---|
| `silu.go` | `SiLUMulRVV(dst, a, b)` | Polynomial SiLU×Mul, 4.1× faster than scalar exp |
| `fastexp.go` | `FastExp(x)` | Schraudolph integer-trick exp, 9.3× faster than math.Exp |
| `quantize.go` | `QuantizeF32RowQ8Block(src, dst)` | Block-scale Q8 quantizer (scalar, not yet RVV asm) |

### `backends/spacemit/k3engine/aipool`

| Function | Notes |
|---|---|
| `GemmQ80x32AIPooledX100Pack` | A100 Q8 GEMM with X100 activation prepack |
| `GemmQ80x32AIPooledGELUX100Pack` | Fused GELU+Q8 activation variant |
| `GemmQ80x32AIPooledSiLUMulX100Pack` | Fused SiLU×Mul+Q8 activation variant |
| `Gemm2Q80x32AIPooledX100PackSameInput` | Dual-GEMM same-input (for gated MLPs) |
| `GemmQ80x32AIPooledGELUX100PackRowScale` | Row-scale GELU variant (Whisper-compatible) |
| A-block buffer pool (`sync.Pool`) | Reduces GC for repeated GEMM calls |

### `backends/spacemit/ime2`

| Function | Notes |
|---|---|
| `PackF32ToQ80x32RowScale` | Row-global scale Q8 packing (Whisper-compatible) |
| `QuantizeF32RowsQ8M4Into` | Allocation-free M4 activation packer |
| `QuantizeF32RowsQ8M4GELUInto` | Fused GELU+Q8 M4 activation packer |
| `QuantizeF32RowsQ8M4GELURowScaleInto` | Row-scale GELU variant |
| `fastTanhQ8`, `geluQ8` | Polynomial tanh/GELU for fused quantizers |

### Shared with Whisper

The A100 row-scale Q8 infrastructure was originally developed for Whisper
(`models/whisper`) and reused for Ideogram4:
- `aipool.GemmQ80x32AIPooled*` helper family
- `ime2.Q80x32` type and packing functions
- `ime2.K3I8I8` / `K3I8I8M4` / `K3I8I8M1` native A100 kernels
- Row-scale vs block-scale Q8 weight packing strategy
- X100 activation prepack pattern
- A100 worker pool with `/proc/set_ai_thread` registration

### Code that should stay model-specific

| File | Reason |
|---|---|
| `k3_fp8_riscv64.go` | Ideogram FP8 E4M3 weight format + cache logic |
| `k3_mlp_riscv64.go` | Ideogram DiT gated MLP (SwiGLU W1/W3/W2) |
| `k3_attention_riscv64.go` | Flash Attention tuned for DiT non-causal shapes |
| `k3_vae_riscv64.go` | Ideogram AutoencoderKLFlux2 VAE layout |
| `k3_prewarm.go` | Ideogram pipeline-specific prewarm traversal |
| `gpu_k3_policy.go` | Ideogram GPU gate interception |
