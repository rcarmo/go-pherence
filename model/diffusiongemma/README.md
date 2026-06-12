# model/diffusiongemma

DiffusionGemma 26B-A4B-it: a block-diffusion text-and-image model with sparse
Mixture-of-Experts (MoE). 26B total parameters, ~4B active per token via top-k
expert selection across 128 experts.

## Model architecture

| Parameter | Value |
|---|---:|
| Architecture | `DiffusionGemmaForBlockDiffusion` |
| Total parameters | 26B |
| Active parameters | ~4B (sparse MoE) |
| Hidden size | 2816 |
| Layers | 30 |
| Attention heads | 16 (8 KV heads, GQA) |
| Head dimension | 256 |
| Vocabulary | 262144 |
| Experts | 128 per MoE layer |
| MoE intermediate | 704 per expert |
| Dense intermediate | 2112 |
| Canvas length | 256 tokens |
| Layer types | sliding_attention (24) + full_attention (6) |
| Max position embeddings | 262144 |

## Weights

### Recommended for K3 (Milk-V Jupiter 2, 31 GB LPDDR)

**RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic** (25.3 GiB)
- Single safetensors file with FP8 E4M3 weights + BF16 norms/scales
- Per-expert tensor format: `experts.{N}.{gate,up,down}_proj.weight`
- Ideal for the A100 Q8 row-scale path (same as Ideogram4 FP8)

**google/diffusiongemma-26B-A4B-it** (48.1 GiB)
- 11-shard BF16 safetensors (original format)
- Fused expert tensors: `experts.gate_up_proj`, `experts.down_proj` as [128, ...]
- Too large for disk unless Ideogram4 weights are removed

### Current status

The FP8 model uses per-expert tensor naming which differs from the original
fused 3D format. The tensor plan builder needs adaptation to accept both formats
before full inference can run.

## Files

| File | Purpose |
|---|---|
| `config.go` | Shape/config parsing from HuggingFace `config.json` |
| `model.go` | Model metadata loading |
| `weights.go` | Safetensors weight loading, tensor binding |
| `tensors.go` | Tensor inventory, readiness checks |
| `text_forward.go` | Text forward plan builder, layer binding |
| `cpu_dispatcher.go` | Full CPU/SIMD forward: attention, MLP, MoE experts, router |
| `gpu_dispatcher.go` | GPU/CUDA dispatcher scaffold with CPU fallback |
| `encoder.go` | Encoder integration |
| `denoiser.go` | Block-diffusion denoiser (not yet implemented) |
| `sampler.go` | Token sampling (top-k, top-p) |
| `chat_prompt.go`, `chat_template.go` | Chat message formatting |
| `vocab.go` | Vocabulary/tokenizer integration |
| `capabilities.go`, `op_status.go` | Runtime capability reporting |
| `ops.go` | Operation plan definitions |
| `processor.go` | Input/output processing |

### K3 / SpacemiT native files

| File | Purpose |
|---|---|
| `k3_dispatcher_riscv64.go` | K3 dispatch hooks: SIMD Sdot, FastExp softmax, RVV SiLU, Saxpy |
| `k3_dispatcher_other.go` | Non-riscv64 stubs |
| `k3_fp8_experts.go` | Per-expert FP8 tensor adapter (in progress) |

## K3 native dispatch

Enable with `GO_PHERENCE_DIFFUSIONGEMMA_K3=1`.

When enabled, hot-path operations are replaced with K3-optimized implementations:

| Operation | Default (scalar) | K3 dispatch |
|---|---|---|
| Q·K dot product | `dot(a, b)` | `simdrt.Sdot(a, b)` — SIMD/RVV vector dot |
| Softmax exp | `math.Exp(...)` | `simdrt.SoftmaxInPlace` with FastExp (9× faster) |
| SiLU activation | `x/(1+exp(-x))` | `rvv.FastSiLU(x)` — polynomial approximation (4× faster) |
| V accumulation | scalar loop | `simdrt.Saxpy(w, v, out)` — SIMD/RVV SAXPY |
| GELUTanhMul | shared SIMD | already uses shared backend |

### Planned K3 optimizations

- A100 Q8 row-scale for dense projections (Q/K/V/O, MLP gate/up/down)
- A100 Q8 for expert projections (128 experts, top-k activated)
- Flash Attention for full_attention layers
- Parallel expert dispatch across X100 workers
- FP8 → Q80x32 resident weight caching with prewarm

## Platform: Milk-V Jupiter 2 / SpacemiT K3

| Feature | Detail |
|---|---|
| SoC | SpacemiT K1 (K3 configuration) |
| X100 cores | 8 general-purpose RISC-V cores (0–7) |
| A100 cores | 8 AI-CPU cores (8–15) with IME2 |
| RAM | 31 GB LPDDR (no swap) |
| RVV | v1.0, 256-bit VLEN |
| Zvfh | FP16 vector compute |
| IME2 | Integer Matrix Extension v2 (A100 only) |
| TCM | 3 MB on-chip SRAM (8 × 384 KB) |

### Memory considerations

- Model weights (FP8): 25.3 GiB
- With mmap, only touched pages are resident
- Sparse MoE: only top-k experts active per token (~4B of 26B)
- Expert weights dominate: 128 experts × 30 layers × {gate, up, down}_proj
- Dense layers are small: 2816 × 2112 per projection
- A100 Q80x32 caches would add ~1.06× the weight size per cached linear

### Reusable infrastructure from Ideogram4

| Backend | Shared code |
|---|---|
| `backends/spacemit/rvv/` | `SiLUMulRVV`, `FastSiLU`, `FastExp`, `F32ToF16RVV` |
| `backends/spacemit/aicpu/aipool/` | A100 Q80x32 GEMM family, activation packing |
| `backends/spacemit/ime2/` | Q80x32 packing, K3I8I8 kernels |
| `backends/simd/runtime/` | `Sdot`, `Saxpy`, `SoftmaxInPlace`, `VecSiLUMul` |
