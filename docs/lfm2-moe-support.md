# LFM2.5-8B-A1B support roadmap

This note tracks initial support planning for Liquid AI's `LiquidAI/LFM2.5-8B-A1B` checkpoint.

## Source reviewed

- Hugging Face checkpoint: <https://huggingface.co/LiquidAI/LFM2.5-8B-A1B>
- `config.json` was inspected from Hugging Face on 2026-05-28.

## Checkpoint summary

```text
model_type: lfm2_moe
architectures: [Lfm2MoeForCausalLM]
dtype: bfloat16
vocab_size: 128000
hidden_size: 2048
num_hidden_layers: 24
num_attention_heads: 32
num_key_value_heads: 8
intermediate_size: 7168
max_position_embeddings: 128000
rope_theta: 5000000
num_experts: 32
num_experts_per_tok: 4
num_dense_layers: 2
moe_intermediate_size: 1792
conv_L_cache: 3
conv_bias: false
use_expert_bias: true
norm_eps: 1e-5
norm_topk_prob: true
routed_scaling_factor: 1.0
tie_word_embeddings: true
```

Layer pattern from the published config:

```text
conv, conv, full_attention,
conv, conv, conv, full_attention,
conv, conv, conv, full_attention,
conv, conv, conv, full_attention,
conv, conv, conv, full_attention,
conv, conv, full_attention,
conv, conv
```

That makes this a hybrid recurrent/convolutional + attention MoE decoder, not a normal Qwen/LLaMA-style dense transformer.

## Fit against current repository

### Reusable now

- `loader/safetensors`, `loader/tokenizer`, and config parsing patterns.
- Existing BF16/F16/F32 loading and NVIDIA BF16 helper paths.
- Qwen3Next work already deals with mixed recurrent/full-attention sequencing and stateful decode, which is conceptually closer to LFM than a plain transformer.
- Qwen MoE work already has router/expert-selection, expert cache, and selected-expert GPU residency patterns.
- Existing layer scheduler/planner work can be reused to reason about dense versus expert residency on constrained VRAM.

### Missing or uncertain

- `model/lfm2` now exists for strict config parsing and tensor-name inventory; runtime block implementations are still pending.
- No native LFM2 convolution/recurrent block implementation exists.
- Tensor names and exact block math need inventory against the checkpoint and, ideally, a Transformers reference trace.
- MoE routing details need parity checks: normalized top-k probabilities, expert bias behavior, routed scaling, and dense-layer exceptions.
- Long-context behavior needs explicit state/cache accounting because the model advertises 128k positions and mixes `conv_L_cache=3` with full-attention layers.

## Proposed package layout

```text
model/lfm2/
  config.go       # lfm2_moe config parser and validation
  tokens.go       # tokenizer/special-token glue if needed
  weights.go      # tensor-name inventory and shape binding
  conv.go         # LFM convolution/state block
  attention.go    # full-attention layer wrapper if not shared
  moe.go          # router, expert selection, expert FFN shapes
  state.go        # conv state + attention KV cache accounting
  generate.go     # CPU/reference generation path
```

CLI:

```text
cmd/lfm2inspect/  # config/tensor/layer-pattern inventory first
```

Do not fold this into `model/qwen`; it should be its own architecture package with only deliberate helper reuse.

## Implementation roadmap

### Phase L0 — Metadata and tensor inventory

- [x] Add `cmd/lfm2inspect` to parse `config.json`, print layer types, MoE settings, conv cache settings, and safetensors tensor groups.
- [x] Add `model/lfm2/config.go` with strict parsing and defaults matching the published config.
- [x] Add tensor-name inventory tests from representative checkpoint names, without committing weights.

Acceptance:

- Inspector identifies `LiquidAI/LFM2.5-8B-A1B` as `lfm2_moe` and reports all critical dimensions.
- Malformed configs fail with explicit errors.

### Phase L1 — Reference fixtures

- [ ] Capture a small Transformers reference for tokenization, first-token logits, one conv layer output, one attention layer output, router top-k, and one MoE expert output.
- [ ] Add tiny JSON/NPY summaries under `testdata/` rather than model payloads.

Acceptance:

- Go config and tensor binding can be checked against deterministic reference metadata.

### Phase L2 — CPU reference path

- [ ] Implement embeddings, RMSNorm, full-attention layers, and tied LM head.
- [ ] Implement LFM convolution/state block with `conv_L_cache=3` semantics.
- [ ] Implement MoE router/top-k/expert FFN with expert-bias and normalized-top-k behavior.
- [ ] Add greedy first-token and short decode parity tests.

Acceptance:

- CPU reference produces matching first token/logit summaries for a fixed prompt.

### Phase L3 — NVIDIA/local optimization

- [ ] Reuse existing NVIDIA BF16/dense matmul paths for attention, router, expert FFNs, and LM head.
- [ ] Reuse or adapt MoE expert-cache policy from Qwen MoE.
- [ ] Use the existing planner concepts to decide which experts/layers stay resident under RTX 3060-class VRAM.
- [ ] Keep conv state on device once the math path is stable.

Acceptance:

- Hardware-gated smoke can generate a short prompt locally with documented residency and throughput.

## Immediate next action

Treat LFM2.5 as a separate roadmap track. After Qwen3-TTS T0/T1 is started, the first concrete LFM task should be `model/lfm2/config.go` plus `cmd/lfm2inspect`, not runtime generation.
