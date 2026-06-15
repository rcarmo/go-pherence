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

Pointer-table fused experts currently require Q4_K gate/up and Q8_0 down. The
Q4_K_M checkpoint uses Q8_0 down experts only on these layers:

```text
0-2,5,8,11,14,17,20,23,26-29
```

Generate a top-6 plan for those compatible layers:

```bash
PLAN=$(scripts/diffusiongemma_active_trace_plan.py \
  /tmp/dg-92-active-trace-top8.log \
  --top 6 \
  --q8-layers '0-2,5,8,11,14,17,20,23,26-29')
```

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
sequential fallback prewarm. Duplicate and invalid IDs are ignored. Layers whose
down experts are not Q8_0 are skipped for pointer prewarm unless
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
Q8_0 down/scatter stage. The partial-resident executor is opt-in:

```bash
export GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_EXPERT_PARTIAL_RESIDENT=1
```

Current exact partial-resident top-N sweep with the same 768MiB expert cache:

| Plan | Encoder MoE | Fused/CPU | Dropped CPU work | Q4 dequant rows | Q8 dequant rows | Notes |
|---:|---:|---:|---:|---:|---:|---|
| top4 | 3.71s | 14/16 | 17223 | 573056 | 456192 | Preserves token but leaves more CPU work. |
| top6 | 3.83s | 14/16 | 15728 | 542080 | 394240 | More resident work, still significant dropped fallback. |
| top8 | 3.63s | 14/16 | 14726 | 508288 | 326656 | Better dropped-work profile. |
| top10 | 2.45s | 14/16 | 14036 | 475904 | 261888 | Best current structural balance. |
| top12 | 2.68s | 13/17 | 14194 | 460416 | 230912 | Loses one fused layer, so not better despite fewer rows. |

Top10 is the current recommended diagnostic plan for this exact partial-resident
profile: it keeps fused coverage at 14 encoder layers while reducing dropped CPU
work the most. The exact-GELU host boundary is measurable but not dominant in the
current profile (roughly a few tenths of a second for encoder fused calls); the
remaining bottleneck is dropped-subset CPU fallback and broad active expert
coverage.

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

- Use the exact partial-resident top10 profile as the current structural target
  for further optimization, not as a default.
- Reduce dropped-subset CPU fallback time or increase resident coverage without
  losing fused layers.
- Reduce Q4_K expert residency footprint or make active-set materialization cheap
  enough that broad active expert coverage no longer falls back to CPU.
- Keep CPU/SIMD fallback faithful to llama.cpp `ggml_mul_mat_id` semantics while
  GPU residency catches up.
