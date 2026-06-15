# DiffusionGemma GGUF GPU bounded profile

This note records the current best **bounded, opt-in** GGUF GPU profile for the
DiffusionGemma Q4_K_M checkpoint. It is not a default: the default path remains
conservative while the GGUF expert residency and fused-kernel work continues.

The profile is meant for RTX 3060-class VRAM constraints and for reproducing the
current 92-token, `canvas=1`, one-token smoke used during the llama.cpp graph
porting work.

## Baseline environment

Use the GGUF checkpoint as the expert/weight source:

```bash
GGUF=/workspace/projects/llama.cpp/models/diffusiongemma-gguf/diffusiongemma-26B-A4B-it-Q4_K_M.gguf
MODEL=./models/diffusiongemma-26B-A4B-it-FP8
PROMPT_IDS=$(seq -s, 100 191)
```

Common bounded profile switches:

```bash
export GO_PHERENCE_DIFFUSIONGEMMA_GGUF_DENSE_TRANSPOSE_CACHE_MB=6144
export GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_LMHEAD_F32_CACHE=1
export GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_CACHE_MB=768
export GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_LAYERS=30
export GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_RESERVE_MB=2048
export GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_CACHE_RESERVE_MB=0
```

The dense transpose and LM-head host caches trade host RAM for removing repeated
host transpose and LM-head setup overhead. The expert cache is still bounded in
VRAM.

## Trace active experts

First collect a trace from a representative prompt. The trace prints the hottest
active experts per layer and marks missing Q4 resident experts with `!`:

```bash
GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_ACTIVE_TRACE_TOP=8 \
./diffusiongemmarun \
  -model "$MODEL" \
  -gguf-model "$GGUF" \
  -gpu-dispatcher \
  -prompt-ids "$PROMPT_IDS" \
  -max-new 1 \
  -canvas 1 \
  -denoise-steps 1 \
  -seed 1 \
  -dispatch-progress \
  -resident-layers 1 \
  -decode 2>&1 | tee /tmp/dg-92-active-trace-top8.log
```

Example trace rows:

```text
gguf_expert_active_trace: layer=0 active=61 work=736 missing_q4=22 missing_q4_bytes=62.4MiB top=69:59,84:44!,89:41!,15:36,35:35,38:35,4:32,7:31
gguf_expert_active_trace: layer=1 active=60 work=736 missing_q4=60 missing_q4_bytes=170.2MiB top=88:80!,35:75!,0:62!,13:60!,14:37!,10:36!,64:35!,40:29!
```

## Build a layer-aware prewarm plan

Pointer-table fused experts currently require Q4_K gate/up and a supported
quantized down path. The Q4_K_M checkpoint uses Q8_0 down experts on these
layers:

```text
0-2,5,8,11,14,17,20,23,26-29
```

Generate a plan for the compatible layers. For the current exact partial-resident
path, use a full active trace and keep at least one resident expert per traced
layer before applying efficiency ordering:

```bash
PLAN=$(scripts/diffusiongemma_active_trace_plan.py \
  /tmp/dg-92-full-active-trace.log \
  --top 128 \
  --q8-layers '0-2,5,8,11,14,17,20,23,26-29' \
  --q5-layers '3-4,6-7,9-10,12-13,15-16,18-19,21-22,24-25' \
  --order efficiency \
  --ensure-layer-coverage \
  --budget-mb 768 \
  --optimize-budget \
  --summary)
```

Older Q8-only top-N plans are still useful for historical comparison, but they do
not cover the Q5_0 down layers that now have pointer-table GPU support.

Then run with the planned prewarm:

```bash
GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_PLAN="$PLAN" \
./diffusiongemmarun \
  -model "$MODEL" \
  -gguf-model "$GGUF" \
  -gpu-dispatcher \
  -prompt-ids "$PROMPT_IDS" \
  -max-new 1 \
  -canvas 1 \
  -denoise-steps 1 \
  -seed 1 \
  -dispatch-progress \
  -resident-layers 1 \
  -decode
```

`GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_PLAN` accepts
`layer:expert,expert;layer:expert,...`. Planned entries are attempted before
sequential fallback prewarm, unless
`GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_PLAN_ONLY=1` is set. Duplicate
and invalid IDs are ignored. Q8_0 and Q5_0 down experts have pointer-table GPU
paths; unsupported down quantization is skipped unless
`GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PREWARM_Q4_ONLY=1` is explicitly set.

## Current measurements

All rows below used the exact 92-token `canvas=1` profile and preserved
`generated=[144]`. Wall-clock values are host-load-sensitive; prefer stable
structural counters such as fused coverage, dropped CPU work, and row counts when
comparing close runs.

### Historical approximate fused-Q4 profile

The original fused Q4_K pointer kernels used tanh-GELU approximation. Those
measurements were useful for residency planning but are no longer the
high-fidelity default:

| Expert profile | Encoder MoE | Notes |
|---|---:|---|
| Sequential layer-0 pointer prewarm | ~3.10s | Full layer-0 prewarm, 128 experts / ~0.62GiB. |
| Trace-derived Q8-compatible top4 plan | 2.205s | Better than sequential, but less coverage than top6. |
| Trace-derived Q8-compatible top6 plan | 2.154s | Best approximate-fused balance in that sweep. |
| Trace-derived Q8-compatible top8 plan | 3.006s | Too much prewarm; worsened CPU fallback/runtime balance. |
| No expert prewarm | 4.463s | More CPU fallback time despite some opportunistic GPU coverage. |

The approximate top6 planned profile prewarmed 154 layer/expert entries,
approximately 0.75GiB, within the 768MiB expert cache. It reduced costly partial
GPU attempts and shifted the hot path toward CPU SIMD fallback for still-missing
experts.

### Exact-GELU partial-resident profile

The current high-fidelity path uses a Q4_K pointer-table dot-only GPU kernel,
then applies exact erf GELU through the explicit exact-GELU boundary before the
down/scatter stage. Down experts support both Q8_0 and Q5_0 pointer-table paths.
The partial-resident executor is opt-in:

```bash
export GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PARTIAL_RESIDENT=1
```

Current exact partial-resident top-N sweep with the same 768MiB expert cache:

| Plan | Encoder MoE | Fused/CPU | Dropped CPU work | Q4 dequant rows | Q8 dequant rows | Q5 dequant rows | Notes |
|---:|---:|---:|---:|---:|---:|---:|---|
| Q8-only top10 (historical) | 2.45s | 14/16 | 14036 | 475904 | 261888 | n/a | Best Q8-only structural balance before Q5 pointer support. |
| all-layer global top10 | load-sensitive | 30/0 | 9209 | 440704 | 459008 | 1346048 | Q5 pointer support active; all encoder layers enter fused dispatch. |
| all-layer efficiency + coverage | load-sensitive | 30/0 | 9172 | 439296 | 470272 | 1331968 | Current 768MiB structural target; +37 kept work items over global top10. |

After Q5_0 pointer support, all-layer plans are the current recommended
diagnostic target for exact partial-resident execution at 768MiB. The latest
layer-coverage-aware efficiency plan preserves `[144]`, reaches `fused=30`, and
reports `partial kept/dropped work=12908/9172`. The exact-GELU host boundary is
measurable but not dominant in the 768MiB profile; the remaining bottleneck is
how many selected experts still land in the dropped CPU subset.

Budget scaling with the same exact partial-resident profile shows the structural
trend clearly even when wall-clock varies with host load:

| Expert cache | Plan source | Prewarmed experts | Kept/dropped work | Q4 dequant rows | Q8 dequant rows | Q5 dequant rows | Notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 768MiB | measured | 168 | 12908/9172 | 439296 | 470272 | 1331968 | Current bounded diagnostic target. |
| 1024MiB | measured | 225 | 15389/6691 | 360448 | 399872 | 1244672 | Optimized plan-only profile; more resident work, still broad dropped subset. |
| 1536MiB | measured | 335 | 18444/3636 | 205568 | 225280 | 1109504 | Much less dropped work; exact-GELU boundary grows. |
| 1792MiB | simulated | 390 | 19338/2742 | n/a | n/a | n/a | Budget simulation only; not yet profiled. |
| 2048MiB | measured | 447 | 19978/2102 | 47872 | 84480 | 934912 | Keeps ~90.5% traced work; remaining fallback mostly Q5_0. |
| 2560MiB | measured | 556 | 20810/1270 | 0 | 0 | 819456 | Q4/Q8 dropped dequant eliminated; remaining fallback is Q5_0. |
| 3072MiB | measured | 667 | 21299/781 | 0 | 0 | 664576 | Keeps ~96.5% traced work; residual fallback is small and Q5_0-only. |

At 2560MiB and above, Q4 and Q8 dropped dequant work are eliminated for this
trace; the remaining measured dropped work is entirely Q5_0. At 3072MiB only 781
of 22080 traced work items remain dropped, so returns from more expert cache are
small. This points toward either smaller Q4_K/Q5_0 resident representation,
better replacement planning under a fixed budget, or a native exact-GELU device
kernel to remove the host boundary as coverage grows.

The plan simulator can model compact resident representations. For example,
current Q4_K resident experts store scale/min values as F32. Modeling those Q4
scale/min values as fp16 with `--q4-scale-bytes 2` shows a structural kept-work
improvement at every tested budget:

| Expert cache | Current Q4 F32 scale/min | Modeled compact Q4 fp16 scale/min |
|---:|---:|---:|
| 768MiB | 12908/22080 | 13900/22080 |
| 1024MiB | 15389/22080 | 16265/22080 |
| 1536MiB | 18444/22080 | 19091/22080 |
| 2048MiB | 19978/22080 | 20422/22080 |
| 2560MiB | 20810/22080 | 21118/22080 |
| 3072MiB | 21299/22080 | 21528/22080 |
| 4096MiB | 21829/22080 | 21943/22080 |
| 5120MiB | 22057/22080 | 22080/22080 |

At the bounded 768MiB target, compact Q4 would keep roughly 992 additional traced
work items. At about 5GiB, compact Q4 would fit the entire traced expert set,
where the current representation still leaves a small tail. That makes a compact
Q4_K resident representation a concrete next kernel/runtime target, provided the
pointer-table kernels preserve llama.cpp/CPU parity.

A direct attempt to store the already-derived per-subblock scale/min values as
fp16 was rejected by a non-skipped CUDA parity check: the half-scale kernel loaded
and ran, but its Q4_K gate/up dot output drifted from the existing F32-derived
scale/min pointer-table path (for example, an up value differed by about
`6.8e-4`). Given the earlier RMSNorm amplification issue, that representation is
not acceptable as the high-fidelity path. The fidelity-preserving compact target
is therefore to store the original Q4_K metadata (`d`, `dmin`, and the 12 packed
scale/min bytes) alongside the packed quants, and reconstruct the F32 scale/min
values inside the pointer-table kernel exactly as `unpackQ4KMatrixRows` does.

## Why this is opt-in

The active experts are prompt- and layer-dependent. The same cache-reserve or
prewarm strategy did not generalize across all tested prompt lengths:

- 768MiB cache + 384MiB prewarm cache reserve helped the 92-token profile but
  worsened 32-token and 128-token profiles.
- 1024MiB cache did not make the reserve setting useful for the 92-token profile.
- Active-set admission caps preserved output but worsened encoder MoE, so they
  were reverted and not committed.

Until a robust automatic heuristic exists, planned prewarm is a diagnostic and
profiling tool, not a production default.

## Next work

- Use the exact partial-resident layer-coverage-aware optimized plan as the
  current structural target for further optimization, not as a default.
- Keep the host-boundary exact-GELU path as the fidelity baseline. A first native
  CUDA erf-approximation attempt produced ~3.6e-4 activation max error versus
  `simd.GELUExactMulTo`, which is too high for DiffusionGemma's high-gain
  post-norm dimensions and was not committed.
- Reduce dropped-subset CPU fallback time or increase resident coverage without
  losing fused layers.
- Reduce Q4_K expert residency footprint or make active-set materialization cheap
  enough that broad active expert coverage no longer falls back to CPU.
- Keep CPU/SIMD fallback faithful to llama.cpp `ggml_mul_mat_id` semantics while
  GPU residency catches up.
