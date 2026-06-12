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

The native text scaffold now accepts both original fused 3D expert tensors and
the FP8 per-expert tensor format. On riscv64/K3, large FP8 projections can be
packed into row-scale Q80x32 tiles and dispatched through the SpacemiT A100
worker pool, with X100 cores packing activations in parallel.

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
| `k3_fp8_experts.go` | Per-expert FP8 tensor adapter and fused/per-expert expert loader |
| `k3_a100_riscv64.go` | K3 A100 row-scale Q80x32 projection cache and GEMM dispatch |
| `k3_a100_experts_riscv64.go` | Per-expert MoE grouping using A100 gate/up/down GEMMs |
| `k3_a100_other.go`, `k3_a100_experts_other.go` | Non-riscv64 stubs |

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

### K3 A100/Q80x32 acceleration

Enable with:

```sh
GO_PHERENCE_DIFFUSIONGEMMA_K3=1 \
GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_Q8=1 \
GO_PHERENCE_DIFFUSIONGEMMA_K3_THREADS=8 \
GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_WORKERS=6
```

Optional prewarm flags move FP8→Q80 packing out of the hot denoising path:

```sh
-k3-q80-prewarm-layers 1              # dense projections for first layer
-k3-q80-prewarm-experts               # also prepack all per-expert tensors; memory-heavy
```

Per-expert tensors are packed in parallel across X100 workers for both
all-expert prewarm and selected on-demand prepack; override the default with
`GO_PHERENCE_DIFFUSIONGEMMA_K3_EXPERT_PREPACK_WORKERS=N`. One-layer all-expert
prewarm currently packs 392 tensors in roughly 8s on the K3 and reports about
865 MB of Q80 cache with zero decoded F32 cache.

The tied BF16 LM head can also use A100 Q80 with
`GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_LMHEAD=1`. It is opt-in because Q80 logits
can drift versus exact BF16/F32 scores, but in the current A100-hidden-state
canvas-4 tail smoke it produced the same generated token while reducing LM-head
time from `5.211s`/2.95 GiB F32 cache to `1.788s`/tiny F32 cache.

TCM staging for Q80 A100 tiles is controlled by `IME2_Q80_TCM`:

- unset / `1`: stage packed A blocks in TCM before `K3I8I8M4` (default)
- `0`: disable Q80 TCM staging
- `ab`: experimental A+B tile staging; correctness-preserving but slower in the
  current DiffusionGemma smoke because repeated B-tile copies dominate

Implemented paths:

- Dense attention projections: Q/K/V are dispatched as same-input A100 GEMMs so
  X100 activation packing is shared.
- Dense MLP: gate/up use a same-input dual GEMM; down uses the row-scale Q80x32
  path.
- Per-expert MoE: positions are grouped by selected expert; each expert batch
  runs A100 gate/up, SIMD GELU×up, then A100 down projection.
- Encoder prompt pass reuses A100 for attention projections and dense MLP.
- FP8 E4M3 weights are decoded with `.weight_scale` and packed into row-scale
  Q80x32 once per process; later calls reuse the cache.

Current smoke numbers on Milk-V/K3, FP8 model, prompt IDs `2,3`, one generated
token, one decoder layer (`-max-dispatch-layers 1`, `-lm-head-top-k 8`):

| Canvas | A100 Q8 | Q80 prewarm | Encoder layer 0 | Decoder layer 0 |
|---:|:---:|:---:|---:|---:|
| 4 | off | — | 1.793s | 12.299s |
| 4 | on | no | 1.574s | 5.833s |
| 16 | off | — | 1.792s | 16.958s |
| 16 | on | no | 1.556s | 6.255s |
| 16 | on | layer 0 + experts | 0.030s | 0.985s |

A 2-layer canvas-16 smoke (`-max-dispatch-layers 2`, `-k3-q80-prewarm-layers 2`,
`-k3-q80-prewarm-experts`) shows the packed-weight path carrying beyond layer 0:

| Mode | Encoder layer 0 | Encoder layer 1 | Decoder layer 0 | Decoder layer 1 |
|---|---:|---:|---:|---:|
| A100 Q8 off | 1.772s | 1.835s | 16.951s | 16.360s |
| A100 Q8 on + Q80 prewarm | 0.031s | 0.031s | 0.985s | 0.987s |

With dense-only Q80 prewarm (`-k3-q80-prewarm-layers 2`) and selected experts
packed on demand, `GO_PHERENCE_DIFFUSIONGEMMA_K3_EXPERT_PREPACK_WORKERS=8`
improved decoder layer times from roughly `5.42s/5.36s` to `3.96s/3.91s`.
Decoder and encoder attention context are split across X100 workers. On a
canvas-64 one-layer decoder smoke this improved layer time from `1.024s`
(`K3_THREADS=1`) to `0.928s` (`K3_THREADS=8`) with identical output; a 32-token
prompt smoke completed cleanly with encoder context parallelism enabled.

The no-A100, A100, and A100+prewarm canvas-16 smoke runs produced the same
sampled output and entropy (`generated=[0]`, accepted canvas tokens `16`, mean
entropy `12.476649250079019`). Remaining high-value work: flash/tiled attention,
selective expert prewarm instead of all-expert prewarm, and broader parity
fixtures.

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
