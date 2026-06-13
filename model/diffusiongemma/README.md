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
| `cpu_dispatcher.go` | Full CPU/SIMD forward: flash/materialized attention, MLP, MoE experts, router |
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

Enable with `-k3` (or `GO_PHERENCE_DIFFUSIONGEMMA_K3=1`).

When enabled, hot-path operations are replaced with K3-optimized implementations.
Set `GO_PHERENCE_DIFFUSIONGEMMA_TIMING=1` to emit per-layer operation timing
similar to the Ideogram4 timing traces. Use `make diffusiongemma-k3-check` for
the fast K3 regression gate (A100 smoke, scaled-FP8 fallback smoke, and affected
Go tests), `make diffusiongemma-k3-smoke` for a
short K3/A100 correctness smoke with expected-output checking
(`DIFFUSIONGEMMA_K3_SMOKE_EXPECT_GENERATED`, default `239683`), and
`make diffusiongemma-k3-profile` or `scripts/diffusiongemma_k3_profile.sh` for a
repeatable full K3 profile run and summary. The profile helper defaults to dense
Q80 residency plus selected expert
retention for all layers (`RETAIN_SELECTED_EXPERT_LAYERS=30`, `SKIP_EVICTION=0`)
as the current practical speed/memory tradeoff on the 31GB K3; set
`SKIP_EVICTION=1` for the max-speed/full-cache profile.

Recommended K3 validation commands:

```sh
make diffusiongemma-k3-smoke     # short A100/Q80 path, expected token check
make diffusiongemma-k3-check     # smoke + FP8 fallback + model-backed Q80/scale tests
make diffusiongemma-k3-profile   # repeatable full profile summary
```

| Operation | Default (scalar) | K3 dispatch |
|---|---|---|
| Q·K dot product | `dot(a, b)` | `simdrt.Sdot(a, b)` — SIMD/RVV vector dot |
| Softmax exp | `math.Exp(...)` | `simdrt.SoftmaxInPlace` with FastExp (9× faster) |
| SiLU activation | `x/(1+exp(-x))` | `rvv.FastSiLU(x)` — polynomial approximation (4× faster) |
| V accumulation | scalar loop | `simdrt.Saxpy(w, v, out)` — SIMD/RVV SAXPY |
| Attention context | materialized score row + softmax | flash-style streaming softmax context, default on |
| GELUTanhMul | shared SIMD | already uses shared backend |

`GO_PHERENCE_DIFFUSIONGEMMA_K3_FLASH_ATTENTION=0` disables the streaming
attention context and restores the materialized score-buffer path for A/B timing
or debugging. The flash path keeps a stable online `(m, l)` softmax state per
query/head and updates the weighted value accumulator directly, avoiding the
score-row allocation and second score pass. The context kernels now batch work
over flattened `(canvas_position, attention_head)` tasks instead of splitting
only by canvas position, so short canvases can still feed all K3/X100 workers
when `positions * heads` is large enough. `attention_flash_test.go` compares
flash and materialized context outputs and includes a forced K3 batched-worker
case (`positions=2`, `heads=16`, `threads=8`) against a serial reference. On a
focused Milk-V/K3 canvas-256, one-layer timing run, the optimized online update
measured attention context at `533ms` versus `610ms` for the materialized path
and reduced layer-0 time from `2.957s` to `2.911s`; full-layer speed remains
dominated by selected-expert Q80 packing.

### K3 A100/Q80x32 acceleration

Enable with `-k3 -k3-a100-q8` plus optional worker/thread flags:

```sh
go run ./cmd/diffusiongemmarun ... -k3 -k3-threads 8 -k3-a100-q8 -k3-a100-workers 6
```

Optional prewarm flags move FP8→Q80 packing out of the hot denoising path:

```sh
-k3-q80-prewarm-layers 1              # dense projections for first layer
-k3-q80-residency-budget-gib 1.0      # choose layer count from Q80 cache budget
-k3-q80-prewarm-experts               # also prepack all per-expert tensors; memory-heavy
```

With a `1.0 GiB` Q80 budget, the FP8 checkpoint currently selects 1 all-expert
layer (`864951296` bytes) or 18 dense-only layers (`1055864832` bytes). Dense
Q80 residency and per-expert Q80 residency are tracked separately, so dense
all-layer prewarm can remain resident without accidentally retaining every
selected expert tensor across nonresident layers.

For multi-step denoising, `-skip-eviction` keeps decoded/Q80 layer caches across
steps. It is memory-heavy but can avoid selected-expert repacking: in a 2-step,
4-layer canvas-16 smoke, second-step layers 1–2 improved from roughly
`477ms/474ms` to `124ms/147ms` with identical output. A bounded middle ground is
`-k3-q80-retain-selected-expert-layers N`, which retains only on-demand selected
expert Q80 caches for the first N layers; with `N=4` the same 4-layer 2-step
smoke kept peak Q80 around `4.17GB` and improved both passes versus evicting
selected experts every layer. The selected-expert prepack helper first filters
already-resident Q80 tensors before scheduling X100 packing workers, so retained
experts skip goroutine setup and repeated build attempts. In a focused
4-layer, 2-step, canvas-16 run with retained selected experts, second-pass
layer 0 hit `prepack=0s` and completed in `45ms`; later second-pass layers
still spent roughly `56-62ms` packing newly selected experts not seen in the
first pass. Selected-expert prefetch remained only a small win in that bounded
case (`8.476s` to `8.412s` wall) and stays opt-in. Worker counts above the
default 8 did not materially improve the residual new-expert pack tail.

Full 30-layer, 2-step, canvas-16 profiles with dense Q80 residency (`2 GiB`
budget selects all dense projections) and A100 LM-head prefetch:

| Mode | Decoder pass 1 | Decoder pass 2 | Max Q80 | Notes |
|---|---:|---:|---:|---|
| bounded, evict selected experts | 16.8s | 15.9s | 2.85GB | safest memory |
| retain selected experts all layers | 12.907s | 8.517s | 12.1GB | 38.073s wall after resident-name filter; no F32 cache growth |
| `-skip-eviction` | 12.929s | 8.407s | 12.1GB | 38.210s wall after resident-name filter; retains all caches |

All three produced the same sampled token in the recorded full-profile runs.

Async Q80 prefetch can interleave packing with compute:

```sh
-k3-q80-prefetch                 # next-layer dense, safer memory
-k3-q80-prefetch-experts         # next-layer all-expert, more memory/bandwidth
-k3-q80-selected-prefetch        # selected experts after router
```

Dense-only next-layer prefetch on a 4-layer canvas-16 smoke improved later
decoder layers from roughly `2.94s/2.90s/2.81s` to `2.75s/2.67s/2.65s`. Full
all-expert next-layer prefetch is not recommended for full passes after the lazy
fallback fixes: it bounds Q80 around ~3.4 GiB but stalls each layer for ~1.8–2.0s
full expert prepack. Selected-expert prefetch reorders router before dense MLP
and packs selected expert weights while dense MLP runs; it modestly improves
later layers (`~642ms→605ms`, `~603ms→594ms`, `~661ms→632ms`) but remains opt-in
because it contends with dense MLP for X100 bandwidth.

Per-expert tensors are packed in parallel across X100 workers for both
all-expert prewarm and selected on-demand prepack; override the default with
`GO_PHERENCE_DIFFUSIONGEMMA_K3_EXPERT_PREPACK_WORKERS=N`. One-layer all-expert
prewarm currently packs 392 tensors in roughly 8s on the K3 and reports about
865 MB of Q80 cache with zero decoded F32 cache.

The tied BF16 LM head can also use A100 Q80 with `-k3-a100-lmhead`. The A100
pass is used as a shortlist generator and candidates are reranked with exact
BF16/F32 dot products (`-k3-a100-lmhead-candidates`, default max(32, 4×topK)).
`-k3-a100-lmhead-prefetch` starts packing the tied embedding Q80 cache while
decoder layers run; after switching exact rerank to BF16×F32 dot, the default
bounded profile keeps LM-head tails around `789–809ms` with identical generated
output. Sparse self-conditioning over top-k logits then
completes in roughly `5ms`; the following step's self-conditioning prefix
(gate/up/down) is batched through the same A100 Q80 path. Self-conditioning
projection Q80 weights are prewarmed with the layer Q80 cache, reducing the
step-2 prefix self-conditioning path from roughly `75ms` to `6ms` in a canvas-16
2-step smoke.

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
  runs A100 gate/up, SIMD GELU×up, then A100 down projection. The generic F32
  fallback uses the same shape instead of looping per-position GEMV: e.g.
  `[16,2816] × [704,2816]^T -> [16,704]` for gate/up and `[16,704] ×
  [2816,704]^T -> [16,2816]` for down.
- Encoder prompt pass reuses A100 for attention projections and dense MLP.
- FP8 E4M3 weights are decoded with `.weight_scale` and packed into row-scale
  Q80x32 once per process; later calls reuse the cache.

### Incremental prompt encoder opportunity

`TextDenoiser` now invalidates encoder KV when the prompt context changes, which
is correct for multi-canvas generation. The next high-impact optimization is an
incremental append encoder: when `PromptIDs` grows by a suffix, previous causal
encoder K/V can be reused because prefix tokens cannot attend to future suffix
tokens. Only the appended suffix needs to pass through the encoder layers, using
previous per-layer K/V as attention prefix and appending newly computed K/V.

Measured cost that motivates this: a two-canvas debug run (`max-new=2`,
`canvas=1`, `max-dispatch-layers=1`) rebuilt the one-layer encoder in ~208ms and
~246ms for prompt lengths 2 and 3. Before respecting `max-dispatch-layers`, the
same debug rebuild cost was ~12–14s because all 30 encoder layers ran. Full
multi-canvas generation will benefit from append-only encoder K/V reuse.

Correctness constraints for implementing it:

- only use append mode when the previous prompt is an exact prefix of the new
  prompt;
- preserve causal masking for suffix tokens: suffix position `i` may attend to
  all prefix K/V and suffix positions `≤ i`, never future suffix positions;
- apply RoPE with absolute positions offset by previous prompt length;
- append K/V per layer and keep deterministic expert order as in the full
  encoder path;
- compare append mode against full encoder rebuild for at least canvas=1 and
  canvas>1 prompts before enabling by default.

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
| A100 Q8 on + Q80 prewarm | 0.031s | 0.031s | 0.081s | 0.068s |

With dense-only Q80 prewarm (`-k3-q80-prewarm-layers 2`) and selected experts
packed on demand, `GO_PHERENCE_DIFFUSIONGEMMA_K3_EXPERT_PREPACK_WORKERS=8`
improved decoder layer times from roughly `5.42s/5.36s` to `3.96s/3.91s`.
Adding a shared FP8 E4M3 decode table reduced the same selected-expert first-use
path further to roughly `2.53s/2.57s` for decoder layers 0/1. A scratch resize
fix also keeps decoder MoE top-k at the configured value for short canvases
(e.g. `16×8=128` assignments instead of accidentally using all 128 experts per
position), reducing resident-weight expert op time from ~304ms to ~44–54ms.
Expert IDs are processed in sorted order so floating-point accumulation is
deterministic across runs.
Decoder and encoder attention context are split across X100 workers. On a
canvas-64 one-layer decoder smoke this improved layer time from `1.024s`
(`K3_THREADS=1`) to `0.928s` (`K3_THREADS=8`) with identical output; a 32-token
prompt smoke completed cleanly with encoder context parallelism enabled. The
largest subsequent win came from lazily loading F32 fallback matrices only when
Q80 dispatch fails: prewarmed canvas-16 decoder layers dropped to roughly
`81ms/68ms`, with `self_attention≈9–10ms`, `dense_mlp≈5ms`, and `experts≈44–55ms`.

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

## OpenAI-compatible resident server

`cmd/diffusiongemmaserver` keeps the DiffusionGemma metadata, weights,
denoiser, tokenizer and K3 caches resident across HTTP requests so sequential
client measurements include realistic API overhead and cache reuse instead of
one-shot process startup.

Example K3 launch for a bounded/full-weight run:

```sh
go run ./cmd/diffusiongemmaserver \
  -model /home/me/models/diffusiongemma-26B-A4B-it-FP8 \
  -listen 127.0.0.1:18080 \
  -allow-slow-cpu -cpu-dispatcher \
  -k3 -k3-a100-q8 \
  -k3-q80-residency-budget-gib 2.0 \
  -k3-q80-retain-selected-expert-layers 30 \
  -k3-q80-selected-prefetch \
  -max-new 1 -canvas 16 -denoise-steps 2
```

Supported endpoints:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/completions`
- `POST /v1/chat/completions`

Requests may use normal OpenAI-style `prompt`/`messages` fields, or
`prompt_ids` for exact token-level benchmarking. Non-streaming responses include
OpenAI-shaped `choices`/`usage` plus `prompt_token_ids`, `generated_token_ids`,
and, when `return_diffusion_steps=true`, a `diffusion_steps` array.

Streaming requests (`stream=true`) emit renderable intermediate canvases as SSE
custom events:

```text
event: diffusion_step
data: {"generated_token_index":0,"step":2,"canvas":[...],"accepted_mask":[true,...],"text":"..."}
```

followed by an OpenAI-style final data chunk and `data: [DONE]`. This lets a
client render the evolving diffusion canvas while preserving compatibility with
clients that only consume the final completion.

Sequential benchmark helper:

```sh
REQUESTS=3 MAX_TOKENS=1 CANVAS=16 DENOISE_STEPS=2 \
  PROMPT_IDS=2,106,107 \
  scripts/diffusiongemma_server_seq_bench.sh
```

Use `STREAM=1` to verify/count intermediate `diffusion_step` events, or set
`SERVER_URL=http://host:port` to benchmark an already-running server.

## llama.cpp PR #24423 reference run

Checked out `ggml-org/llama.cpp` PR `#24423` at `4a6735f1` into
`/home/me/src/llama.cpp-pr24423` and downloaded
`unsloth/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf`
under `/home/me/models/diffusiongemma-26B-A4B-it-GGUF/`.

Reference command used on the Milk-V K3:

```sh
build/bin/llama-diffusion-cli \
  -m /home/me/models/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
  -ngl 99 \
  --n-cpu-moe 20 \
  --diffusion-steps 256 \
  -n 1 \
  -p 'Say hello briefly.'
```

Observed caveats/results on this board:

- llama.cpp warned `no usable GPU found`, so `-ngl 99` was accepted but ignored
  by this CPU-only riscv64 build.
- `--n-cpu-moe 20` was accepted and maps to the PR's CPU-MoE offload flag.
- The generic diffusion parameter logged `steps=256`, but the DiffusionGemma
  entropy-bound runner used `max_steps=48` from `diffusion.eb_max_steps` GGUF
  metadata/defaults and early-stopped after 11 steps for the single 256-token
  canvas block.
- Output sample: `Hello!`
- Runtime: `529286.45ms`; reported `time per step: 48116.95ms`, `11 steps over
  1 blocks`, `throughput: 0.5 tok/s (256 tok in 529286.45ms)`, and `in-step
  parallel 5 tok/s`.

Strategies in PR #24423 that map directly to go-pherence/K3 work:

1. Keep model/server resident for sequential requests.
2. Stream renderable per-step canvas frames instead of shipping full logits.
3. Keep self-conditioning and sampling reductions as close to the device/backend
   as possible.
4. Track effective block/canvas/denoising-step counts and split model throughput
   from API/rendering overhead.
5. Use explicit MoE/cache placement policy knobs (`--n-cpu-moe` there;
   selected-expert Q80 retention and Q80 residency budgets here).

The resident OpenAI-compatible server now exposes `diffusion_stats` in every
final response/chunk so go-pherence measurements report actual generated tokens,
blocks, executed denoising steps, canvas-position evaluations, generated tok/s,
and canvas-position tok/s. This mirrors the PR's later `STATS` commits and makes
future runner-vs-server comparisons unambiguous.

### Canonical K3 256-canvas benchmark

Use `scripts/diffusiongemma_k3_canvas256_bench.sh` for apples-to-apples K3
comparison runs. It refuses `PROMPT_IDS`, sends the text/chat prompt through the
server tokenizer, defaults to `canvas=256`, sends/reports the llama.cpp public
`REQUESTED_DIFFUSION_STEPS=256` as `diffusion_steps`, and keeps that separate
from effective `DENOISE_STEPS`/`effective_denoising_steps`. This mirrors the
PR #24423 distinction where `--diffusion-steps 256` was accepted but the
DiffusionGemma entropy-bound path used metadata/default `max_steps=48` and the
reference prompt stopped after 11 steps. The harness uses the go-pherence K3
sparse LM-head preset (`-k3-a100-lmhead -k3-a100-lmhead-prefetch -lm-head-top-k
64`) so the run does not accidentally fall back to a full `256 × vocab × hidden`
LM-head pass. Override `DENOISE_STEPS` when comparing an observed llama.cpp
effective step count. Set `SAMPLER_MODE=entropy_bound` to use the llama.cpp-style
sampled-token entropy-bound accept/renoise loop; the default remains `argmax`
for existing fast/debug runs.

### Sampler modes

`DenoisingConfig.Sampler.Mode` selects the canvas update rule:

- `argmax` — legacy go-pherence behavior: replace the full canvas with argmax
  predictions each denoising step.
- `entropy_bound` — llama.cpp-style correctness mode: pre-draw per-position
  sampling and renoise tokens, sample from the temperature-scaled logits, accept
  the low-entropy prefix using the entropy-bound rule, and renoise rejected
  positions.

The CLI exposes `-sampler-mode argmax|entropy_bound`; the OpenAI-compatible
server accepts flat JSON `sampler_mode` or a structured `denoising.sampler.mode`.

### Self-conditioning parity

llama.cpp PR #24423 uploads the previous step's raw logits as an input tensor
`[n_vocab, canvas]`, then computes `softmax(logits * sc_temp_inv) @ embed_tokens`,
scales by `sqrt(hidden)`, and feeds that soft embedding through the
self-conditioning MLP. go-pherence keeps the raw logits in process and computes
the same soft embedding directly before the existing K3/Q80 self-conditioning
MLP; this avoids a separate raw-logit upload buffer but preserves the reference
math. `self_conditioning_test.go` locks the raw-logit softmax-matmul behavior,
including sparse `-Inf`/`NaN` handling used by sparse LM-head outputs.
