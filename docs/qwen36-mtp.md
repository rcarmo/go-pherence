# Qwen3.6 27B Native MTP roadmap

## Active goal

Qwen3.6 27B native MTP is the main Qwen stress target. Current day-to-day MTP algorithm work should use the Gemma4 E4B pair because it fits fully on the local RTX 3060 and gives second-scale real-prompt smokes; Qwen3.6 remains the native/in-checkpoint MTP path to bring up after the base architecture and quantized loader path are reliable. Keep this separate from the Gemma4 assistant-drafter MTP path and the Orthrus-inspired stock-weight speculative scaffold.

Definition of done for the first useful milestone:

- load Qwen3.6 text config and tensors far enough to reject/route unsupported pieces clearly;
- support the base Qwen3.6 text forward path needed by the 27B checkpoint;
- load the native `mtp.*` head metadata/weights;
- run greedy CPU correctness for at least one short prompt;
- run `speccheck` normal-vs-native-MTP parity with K=1;
- only then optimize NVIDIA/KV reuse.

## Public checkpoint finding

Hugging Face search shows active Qwen3.6 27B MTP artifacts, including:

- `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
- `unsloth/Qwen3.6-27B-MTP-GGUF`
- `havenoammo/Qwen3.6-27B-MTP-UD-GGUF`
- `froggeric/Qwen3.6-27B-MTP-GGUF`

The originally inspected safetensors checkpoint is `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`.

Top-level config:

```text
architectures: [Qwen3_5ForConditionalGeneration]
model_type: qwen3_5
language_model_only: true
text_config.model_type: qwen3_5_text
```

Text config essentials:

```text
hidden_size: 5120
num_hidden_layers: 64
num_attention_heads: 24
num_key_value_heads: 4
head_dim: 256
vocab_size: 248320
layer_types: linear_attention x3, full_attention x1, repeated
mtp_num_hidden_layers: 1
mtp_use_dedicated_embeddings: false
```

Native MTP tensor prefix:

```text
mtp.fc.weight                              BF16 [5120, 10240]
mtp.pre_fc_norm_embedding.weight           BF16 [5120]
mtp.pre_fc_norm_hidden.weight              BF16 [5120]
mtp.layers.0.input_layernorm.weight        BF16 [5120]
mtp.layers.0.self_attn.q_proj.weight       BF16 [12288, 5120]
mtp.layers.0.self_attn.k_proj.weight       BF16 [1024, 5120]
mtp.layers.0.self_attn.v_proj.weight       BF16 [1024, 5120]
mtp.layers.0.self_attn.o_proj.weight       BF16 [5120, 6144]
mtp.layers.0.self_attn.q_norm.weight       BF16 [256]
mtp.layers.0.self_attn.k_norm.weight       BF16 [256]
mtp.layers.0.post_attention_layernorm.weight BF16 [5120]
mtp.layers.0.mlp.gate_proj.weight          BF16 [17408, 5120]
mtp.layers.0.mlp.up_proj.weight            BF16 [17408, 5120]
mtp.layers.0.mlp.down_proj.weight          BF16 [5120, 17408]
mtp.norm.weight                            BF16 [5120]
```

This is not the Gemma4 LiteRT q-only assistant layout. It is an in-model one-layer native MTP head with its own full Q/K/V attention and MLP, plus a fusion/projection matrix over embedding + hidden (`mtp.fc.weight`).

## MLX 4-bit candidates

Later HF search found MLX 4-bit native-MTP Qwen3.6 candidates that avoid the original NVFP4 public-loading blocker:

| Model | Size | Type | Hidden | Layers | MTP layers | Notes |
|---|---:|---|---:|---:|---:|---|
| `samwang0041/Qwen3.6-27B-MLX-4bit-MTP` | ~15.37GB | `qwen3_5` | 5120 | 64 | 1 | Best dense Qwen candidate found; MLX affine 4-bit group 64. |
| `kradih/Qwen3.6-27B-MTP-4bit-MLX` | ~15.37GB | `qwen3_5` | 5120 | 64 | 1 | Similar/duplicate-looking candidate. |
| `stamsam/Qwen3.6-35B-A3B-Claude-4.7-Opus-Reasoning-Distilled-MLX-oQ4-MTP` | ~20.0GB | `qwen3_5_moe` | 2048 | 40 | 1 | MoE; likely more loader/runtime complexity. |
| `m5max/Huihui-Qwen3.6-35B-A3B-Claude-4.6-Opus-abliterated-mlx-oQ8-mtp` | ~37.7GB | `qwen3_5_moe` | 2048 | 40 | 1 | 8-bit MoE; not suitable for current hardware. |

The recommended Qwen next target is `samwang0041/Qwen3.6-27B-MLX-4bit-MTP`: dense, native MTP, MLX affine 4-bit. It is downloaded locally as `models/qwen3.6-27b-mlx4-mtp` (~15GB) and is ignored by git. It is still much larger than Gemma4 E4B and unlikely to fit fully on the RTX 3060, but it avoids the NVFP4 gate and is a better stress target than the original NVFP4 checkpoint.

Local metadata status:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwenmtpmeta \
  -model models/qwen3.6-27b-mlx4-mtp \
  -strict
```

This now passes after accepting `language_model.mtp.*` tensor names. The checkpoint has 29 native-MTP tensor entries, including MLX packed triples for projection matrices, and `missing_mtp_tensor_count=0`.

Current loader status:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen36run \
  -model models/qwen3.6-27b-mlx4-mtp \
  -prompt "Hello" -steps 1 -mtp -mtp-steps 1
```

The loader reaches the base linear-attention weights and then fails because Qwen3.5/Qwen3.6 base loading still expects BF16/NVFP4 for that path:

```text
language_model.model.layers.0.linear_attn.in_proj_qkv.weight: safetensors: unsupported dtype "U32"
NVFP4 fallback: ... dtype=U32, want U8 NVFP4
```

Implemented local smoke step: Qwen3.5/Qwen3.6 base layers and native-MTP head now load MLX affine U32 packed weights through `backends/mlx` instead of requiring BF16/NVFP4. `qwen36run` also handles MLX-packed embeddings and LM head for this checkpoint.

Current CPU smoke:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen36run \
  -model models/qwen3.6-27b-mlx4-mtp \
  -prompt "Hello" -steps 1 -mtp -mtp-steps 1
```

Latest local result:

```text
passed: true
next_id: 119
mtp_next_id: 220
mtp_accepted_by_greedy: false
duration_ms: 30860
tokens_per_second: 0.0648
```

This proves the dense MLX checkpoint can load and run the base+native-MTP diagnostic path on CPU.

Initial NVIDIA status:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen36run \
  -gpu -gpu-prewarm=false -gpu-lm-head=false -gpu-timing \
  -model models/qwen3.6-27b-mlx4-mtp \
  -prompt "Hello" -steps 1 -mtp -mtp-steps 1
```

Latest local result:

```text
passed: true
next_id: 132386
mtp_output_len: 5120
linear_stats.calls: 489
linear_stats.gpu_calls: 8
linear_stats.cpu_calls: 481
duration_ms: 39530
```

The NVIDIA MLX path now uses an MLX-specific GPU weight cache under the existing `-gpu-cache-mb` budget. MLX cache admission is intentionally no-evict: the full 27B working set is larger than local VRAM, so evicting resident weights for one-off later-layer uploads caused zero hits on the next token. The cache now preserves the resident prefix and lets overflow weights fall back to CPU.

Latest local one-step MTP cache smoke:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen36run \
  -gpu -gpu-prewarm=false -gpu-lm-head=false -gpu-timing \
  -gpu-cache-mb 11000 \
  -model models/qwen3.6-27b-mlx4-mtp \
  -prompt "Hello" -steps 1 -mtp -mtp-steps 1
```

```text
passed: true
next_id: 119
mtp_next_id: 220
linear_stats.calls: 489
linear_stats.gpu_calls: 144
linear_stats.cpu_calls: 345
gpu_cache.entries: 144
gpu_cache.uploads: 144
gpu_cache.used_bytes: 11386587136
duration_ms: 25695
```

Two-step cache reuse smoke:

```text
steps: 2
linear_stats.calls: 978
linear_stats.gpu_calls: 288
linear_stats.cpu_calls: 690
gpu_cache.hits: 144
duration_ms: 44566
```

Follow-up improvements landed:

- persistent NVIDIA input/output scratch for Qwen MLX GEMV, avoiding per-call DevBuf allocation;
- MLX prewarm through the existing `-gpu-prewarm` path;
- native-only MLX GPU upload for Qwen (`UploadMLXWeightNative`) instead of storing both native and GPTQ-compatible representations.

Latest local one-step MTP smoke with native-only cached MLX:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen36run \
  -gpu -gpu-prewarm=true -gpu-lm-head=false -gpu-timing \
  -gpu-cache-mb 11000 \
  -model models/qwen3.6-27b-mlx4-mtp \
  -prompt "Hello" -steps 1 -mtp -mtp-steps 1
```

```text
passed: true
next_id: 119
mtp_next_id: 220
linear_stats.calls: 489
linear_stats.gpu_calls: 393
linear_stats.cpu_calls: 96
gpu_cache.entries: 393
gpu_cache.hits: 393
gpu_cache.used_bytes: 11250892800
prewarm_ms: 1005
duration_ms: 9360
```

Two-step reuse smoke:

```text
steps: 2
linear_stats.calls: 978
linear_stats.gpu_calls: 786
linear_stats.cpu_calls: 192
gpu_cache.hits: 786
duration_ms: 19834
tokens_per_second: 0.151
```

This is a substantial improvement over CPU-only (`~30.9s` for the one-step MTP smoke) and over the previous mixed GPU/CPU cache path (`~25.7s`). Follow-up update: `qwen36run` now uses the GPU cache for MLX LM-head logits when `-gpu-lm-head=true`, and Qwen MLX prewarm uses an explicit decode-hot layer-prefix placement policy rather than size-sorted/checkpoint-order cache churn. Size-only sorting increased resident entry count but scattered layers and was slower; preserving a stable decode prefix gives deterministic cache hits across tokens.

Current validation command:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen36run \
  -gpu -gpu-prewarm=true -gpu-lm-head=true \
  -gpu-cache-mb 10600 \
  -model models/qwen3.6-27b-mlx4-mtp \
  -prompt "Hello" -steps 1 -mtp -mtp-steps 1
```

Latest result:

```text
passed: true
duration_ms: 4063
next_id: 119
mtp_next_id: 220
linear_stats.gpu_calls: 489
linear_stats.cpu_calls: 0
gpu_cache.entries: 349
gpu_cache.hits: 349
gpu_cache.transient_uploads: 282
lm_head_stats.gpu_ms: 9
```

The key placement lesson is that the cache must leave VRAM headroom for transient native MLX uploads. A too-large resident cache (`~11GB`) prevents overflow weights from reaching CUDA and causes CPU fallback. Around `10600MB` keeps a useful resident prefix while leaving enough scratch for transient overflow GEMVs, so all 489 linear calls dispatch through NVIDIA on the local RTX 3060.

Remaining performance gaps are no longer missing NVIDIA plumbing: they are transient upload volume, lack of GPU-side argmax/top-k for logits, and broader layer-streaming/KV reuse policy. `qwen36run` exposes `-gpu-mlx-overflow` to control transient native-MLX uploads for overflow weights; on the local RTX 3060 the best current setting keeps it enabled with `-gpu-cache-mb 10600`, yielding no CPU GEMV fallback for the one-step MTP smoke.

Qwen prompt-state reuse is now available in-process and searches for the longest cached prefix, not just exact prompts:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen36run \
  -gpu -gpu-prewarm=true -gpu-lm-head=true -gpu-cache-mb 10600 \
  -kv-reuse -kv-repeat 2 -kv-chunk-size 32 \
  -model models/qwen3.6-27b-mlx4-mtp \
  -prompt "Hello" -steps 1 -mtp -mtp-steps 1
```

Latest smoke:

```text
passed: true
kv_reuse: true
kv_cache_hit: true
kv_repeat: 2
duration_ms: 5676
linear_stats.gpu_calls: 489
linear_stats.cpu_calls: 0
```

The cached state includes the full `Qwen35BaseForwardState`, so it covers both full-attention K/V and linear-attention recurrent state. The lookup walks from the full prompt down to earlier chunk boundaries and restores the longest matching prefix. A validation smoke with `-kv-chunk-size 2` reports `kv_reused_tokens=3` and `kv_stored_chunks=2` for a three-token prompt repeated in-process, proving chunk-boundary state storage, cached-prefix restore, and suffix skip are wired. Unit tests verify that stored `Qwen35BaseForwardState` snapshots are cloned/restored without aliasing. Cross-request persistence/page-offload remains future work.

Layer-streamed prompt prefill is now available for Qwen diagnostics:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen36run \
  -gpu -gpu-prewarm=true -gpu-lm-head=true -gpu-cache-mb 10600 \
  -layer-streamed-prefill -prefill-chunk-size 4 \
  -model models/qwen3.6-27b-mlx4-mtp \
  -prompt "Hello world again" -steps 1 -mtp -mtp-steps 1
```

The scheduler processes prompt chunks layer-by-layer: for a chunk, it keeps a layer's weights hot, runs that layer across every token in the chunk while updating full-attention K/V or linear recurrent state in token order, then moves to the next layer. A comparison against the sequential path for `Hello world again` matched `next_id`, `mtp_next_id`, `hidden_abs_sum`, and `mtp_abs_sum`, validating the layer-streamed state update order.

MTP verifier state and accepted-prefix commit are now combined with the restored/streamed prompt state. `qwen36run` compares drafted IDs with verifier IDs, appends accepted draft tokens, appends the LiteRT-style verifier bonus token, advances a cloned verifier runner through the committed tokens, and then commits the resulting `Qwen35BaseForwardState` back into the main runner. The report exposes `mtp_committed_tokens` and `mtp_commit_state_pos`.

`qwen36run -mtp-generate` now runs a repeated native-MTP generation loop:

1. draft up to `-mtp-steps` tokens from the current verifier hidden state;
2. verify drafts against the main model;
3. commit accepted prefix;
4. commit verifier bonus token on mismatch/all-accepted completion;
5. repeat until `-steps` output tokens are produced.

Validation command:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen36run \
  -gpu -gpu-prewarm=true -gpu-lm-head=true -gpu-cache-mb 10600 \
  -kv-reuse -kv-chunk-size 2 \
  -layer-streamed-prefill -prefill-chunk-size 4 \
  -model models/qwen3.6-27b-mlx4-mtp \
  -prompt "Hello world again" \
  -steps 3 -mtp -mtp-generate -mtp-steps 2
```

Latest smoke:

```text
passed: true
mtp_generate: true
generated_ids: [119, 119, 119]
mtp_generated_accepted: 0
next_id: 119
linear_stats.gpu_calls: 2976
linear_stats.cpu_calls: 0
```

The loop is functional, but this prompt/model state currently accepts no draft tokens, so it emits verifier bonus tokens. Acceptance quality is now the next correctness target.

MTP audit notes:

- The generation loop now treats the first verifier token after prompt prefill as a bonus token, commits it first, then seeds MTP with that committed token and its pre-norm hidden row. This matches the LiteRT-style sequence more closely than comparing the first MTP draft directly against the prompt's greedy token.
- Position `pos` vs `pos+1` was spot-checked for the MTP draft layer and did not change the observed draft IDs on the local smoke.
- Greedy-seed diagnostic also remains rejected: for `Hello world again`, base verifier token is `119`, prefill MTP predicts `220`, greedy-seed MTP predicts `13`, and post-commit MTP predicts `13` while verifier remains `119`.
- The native-MTP draft cache is now seeded with prompt MTP K/V for prompt positions before the current seed token. This makes the draft layer's self-attention context match the prompt prefix instead of starting from an empty one.
- The Qwen3Next linear-attention recurrent step now follows the reference layout more closely: raw `in_proj_qkv` feeds causal conv as `[q,k,v]`, q/k are repeated to value heads after the conv, recurrent state is `[value_head, value_dim, key_dim]` rather than including an extra key-head axis, and the recurrent update uses the gated-delta residual form `state = state * exp(g) + beta * (v - state·k) * k` with `q/sqrt(key_dim)` readout.
- Qwen3Next gated RMSNorm is now applied per value head (`head_v_dim`) before multiplying by `silu(z)`; the previous whole-vector RMSNorm call was a no-op because the norm weight is only one value-head wide.
- Full-attention `q_proj` output is now split per head as `[q_head, gate_head]`, matching the Transformers reference. The previous flat `[all_q, all_gate]` split was a major verifier-parity bug and also affected the native-MTP layer.
- After these fixes, semantic output improves substantially (`The capital of France is` decodes as ` known for its rich history` in the local 5-token smoke), and native-MTP acceptance is no longer stuck at zero. A local `Hello world again` smoke with `-mtp-generate -mtp-steps 2` accepted one draft token and produced `! I'm back again`.
- The generation loop now keeps the native-MTP self-attention cache current across accepted-prefix/bonus-token commits. It seeds MTP with the full prompt, commits the MTP K/V for the seed token even when a draft is rejected, and commits additional MTP K/V rows for accepted draft steps. A 10-token `Hello world again` smoke produced `! I'm back again with another post about my` with `mtp_generated_accepted=3` and `decode_tokens_per_second≈0.67` on the local RTX 3060 path.
- Generation reports now include `mtp_generated_drafted`, `mtp_generated_rounds`, `mtp_generated_bonus_tokens`, and `mtp_generated_acceptance_rate` (accepted drafts divided by drafted candidates). A 12-token smoke produced `! I'm back again with another post about my journey into`, with `mtp_generated_accepted=5`, `mtp_generated_drafted=14`, `mtp_generated_rounds=7`, `mtp_generated_bonus_tokens=7`, and acceptance rate ≈35.7%.
- `-mtp-generate` no longer runs the one-shot diagnostic pass afterward, so generation reports now reflect the generation loop itself. On the local 12-token `Hello world again` smoke, sequential decode remains faster (~0.81 tok/s) than native MTP (~0.68 tok/s for `-mtp-steps 2`, ~0.72 tok/s for `-mtp-steps 1`) because each draft candidate still needs an LM-head argmax and verifier commit is serial. This identifies GPU-side argmax/top-k and batched verifier/draft evaluation as the next performance targets.
- An experimental NVIDIA `ArgmaxF32` kernel and MLX LM-head argmax path are available behind `GO_PHERENCE_NVIDIA_ARGMAX=1`, but it remains slower than downloading logits on the local RTX 3060, so it is disabled by default. The implementation now uses a hierarchical per-block GPU reduction plus host reduction of partial winners, but the current PTX/launch path is still not competitive. The default path remains the known-good GPU MLX GEMV plus host argmax/download path until a better fused GEMV+top-k or CUB-style reduction replaces it.
- `qwen36run -mtp-adaptive` can now fall back to plain verifier decode when accepted/drafted ratio remains below `-mtp-min-acceptance` after `-mtp-warmup-rounds`. On the local 12-token `Hello world again` smoke with `-mtp-steps 1 -mtp-adaptive -mtp-min-acceptance 0.75 -mtp-warmup-rounds 4`, it fell back after 4 MTP rounds (`accepted=2`, `drafted=4`, rate=50%) and produced the same text at ~0.73 tok/s. This is still slower than pure sequential (~0.81 tok/s) for short outputs, but bounds downside when MTP acceptance is low. Unit coverage now locks the fallback policy for disabled mode, warmup gating, threshold equality, and zero-draft edge cases.
- Focused verifier-parity tests now lock in the Qwen3Next layout fixes: full-attention per-head `[q_head, gate_head]` split, raw linear-attention `[q,k,v]` split plus q/k repeat to value heads, per-value-head recurrent-state sizing, and per-value-head gated RMSNorm.
- `qwen36run -mtp-verify-chunk` adds an experimental speculative-verifier chunk path that precomputes verifier states/IDs for each draft chunk from a cloned verifier state, then commits only the accepted prefix and bonus. It currently preserves output/acceptance but is slower than the serial verifier path (`~0.42 tok/s` vs `~0.69 tok/s` for the 12-token local smoke) because the chunk path still runs per-token verifier steps internally. It is disabled by default and retained as scaffolding for true layer-batched verification.
- `qwen36run -mtp-verify-layer-chunk` adds a second experimental comparison path that computes verifier IDs for a whole draft chunk through `ForwardChunkLayerStreamedDetailed`. The detailed model API returns exact per-prefix `Qwen35BaseForwardState` snapshots assembled during the single layer-major pass, so both whole-chunk and partial accepted prefixes can commit directly without rerunning accepted tokens. It preserves output/acceptance (`mtp_verifier_layer_chunks=7` on the local 12-token smoke) and improves the experimental path slightly (`~0.43 tok/s`), but remains slower than serial verification because verifier comparison is still an extra pass rather than replacing serial verification in the default path.
- `qwen36run -compare-sequential` now runs an apples-to-apples sequential baseline from the same post-prefill state after an MTP generation run and reports `sequential_decoded`, `sequential_duration_ms`, `sequential_decode_tokens_per_second`, `sequential_linear_stats`, `sequential_lm_head_stats`, and `mtp_speedup_vs_sequential`. The normal `linear_stats` and `lm_head_stats` are frozen before the comparison run, so they describe the MTP generation only. An 8-token `Hello world again` comparison produced identical text, with native MTP at ~0.61× sequential speed under current settings.
- Native-MTP generation now caps draft work to remaining output capacity and skips draft work entirely when only one output slot remains, emitting the current verifier bonus token directly. Unit coverage locks this tail-work policy.
- `qwen36run -kv-cache-mb` makes the in-process prompt-state cache budget configurable instead of fixed at 2GiB. Reports now include `kv_cache_max_bytes`, `kv_cache_used_bytes`, and `kv_cache_entries`. Cache accounting now includes the full stored `Qwen35BaseForwardState` sidecar (full-attention K/V, linear recurrent state, hidden, and pre-norm hidden) rather than only the hidden vector. A 512MiB-cache smoke with `-kv-reuse -kv-repeat 2 -kv-chunk-size 2` restored 3 prompt tokens from two chunk entries using ~318MiB; a 1MiB cache correctly evicts the oversized states. Sidecar prompt-state snapshots are pruned whenever the byte-budget LRU evicts entries, and `kv_stored_chunks` now counts only stores that actually survive the cache budget. Reports now also expose `kv_prefill_tokens`, `kv_suffix_tokens`, `kv_skipped_prefill_tokens`, and `kv_reuse_efficiency`; the two-repeat smoke restored 3 tokens, prefilling only 3 of 6 total prompt-token visits for 50% reuse efficiency.
- `qwen36run -kv-prime-prompt` primes the prompt-state cache with one prompt before running the main prompt, enabling cross-prompt longest-prefix diagnostics. A smoke with `-kv-prime-prompt "Hello world" -prompt "Hello world again" -kv-chunk-size 2` restored the 2-token prefix, prefilling only the 1-token suffix and reporting ~66.7% reuse efficiency. Unit coverage locks primed-prefix restoration for extended prompts and unrelated-prompt misses.
- `qwen36run -kv-compare-cold` runs a cold prefill comparison for the same main prompt after the cached run and reports `kv_cached_duration_ms`, `kv_cold_duration_ms`, `kv_speedup_vs_cold`, `kv_cached_next_id`, `kv_cold_next_id`, and `kv_cold_matches_cached`. The timing now measures prompt prefill only, not decode/prime overhead. The `Hello world` → `Hello world again` smoke measured ~2.58× cached-vs-cold speedup for the main prompt path after restoring the 2-token prefix, with cached/cold prefill parity.
- The first GPU hot-tier building block is available as `Qwen35GPUForwardState` plus `UploadQwen35ForwardStateGPU`, `DownloadQwen35ForwardStateGPU`, and `Free`. It mirrors full-attention K/V and linear recurrent Conv/SSM state into NVIDIA buffers with byte accounting. Current use is upload/download parity only; CPU prompt-state cache remains the source of truth until inference can consume the GPU buffers directly.
- KV reuse reports now include lookup/store counters: `kv_lookup_attempts`, `kv_lookup_hits`, `kv_lookup_misses`, `kv_store_attempts`, and `kv_evicted_stores`, alongside retained `kv_stored_chunks`. This makes cache-budget effects and actual retained stores explicit in long-running diagnostics.
- Reports include a compact `summary` object with the headline dashboard fields (`kv_hit`, `kv_reuse_efficiency`, `kv_speedup_vs_cold`, `mtp_acceptance_rate`, `mtp_speedup_vs_sequential`, and `decode_tokens_per_second`) so scripts can consume one stable block instead of recomputing derived values from the detailed counters.

Validation command:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen36run \
  -gpu -gpu-prewarm=true -gpu-lm-head=true -gpu-cache-mb 10600 \
  -kv-reuse -kv-repeat 2 -kv-chunk-size 2 \
  -layer-streamed-prefill -prefill-chunk-size 4 \
  -model models/qwen3.6-27b-mlx4-mtp \
  -prompt "Hello world again" -steps 1 -mtp -mtp-steps 2
```

Latest smoke:

```text
passed: true
kv_cache_hit: true
kv_reused_tokens: 3
mtp_draft_ids: [13, 13]
mtp_verifier_ids: [119]
mtp_committed_tokens: [119]
mtp_commit_state_pos: 4
linear_stats.gpu_calls: 1467
linear_stats.cpu_calls: 0
```

## Important blocker for the original NVFP4 checkpoint

The originally inspected safetensors checkpoint is NVFP4:

```text
quantization_config: compressed-tensors/modelopt-style FP4/NVFP4 groups
```

Current go-pherence intentionally rejects public NVFP4 loading/generation until real CPU/NVIDIA logits/tokens agree. Since MLX 4-bit Qwen3.6 native-MTP checkpoints now exist, the fastest practical route is to try one of those before spending more time on public NVFP4 generation.


## llama.cpp `mtp-clean` reference mapping

Reference branch: <https://github.com/am17an/llama.cpp/tree/mtp-clean> at inspected commit `2dff7ff`.

Important implementation points to port/adapt:

### Conversion / tensor naming

`conversion/qwen.py` defines `_Qwen35MtpMixin`:

- extends `block_count` by `mtp_num_hidden_layers`;
- writes `nextn_predict_layers`;
- can split trunk and MTP with `--no-mtp` / `--mtp`;
- remaps HF native MTP tensors:

```text
mtp.layers.0.*                  -> model.layers.{num_hidden_layers}.
mtp.fc.weight                   -> model.layers.{bid}.eh_proj.weight
mtp.pre_fc_norm_embedding.weight -> model.layers.{bid}.enorm.weight
mtp.pre_fc_norm_hidden.weight    -> model.layers.{bid}.hnorm.weight
mtp.norm.weight                  -> model.layers.{bid}.shared_head.norm.weight
```

For go-pherence we can either keep the HF `mtp.*` names directly or mirror this logical mapping internally. Keeping direct HF names is less invasive for safetensors.

### Base Qwen3.5/Qwen3.6 text model

`src/models/qwen35.cpp` is the closest architecture reference. Key points:

- `nextn_predict_layers` are treated as extra decoder blocks appended after the main stack.
- Main forward executes only `n_layer - nextn_predict_layers`. Added metadata helpers to compute main-layer count and classify main full/linear-attention layers vs appended native-MTP layers.
- Recurrent/linear-attention layers are all non-full-attention layers before the MTP tail:

```cpp
n_main = n_layer - nextn_predict_layers
recurrent[i] = i < n_main && ((i + 1) % full_attention_interval != 0)
```

- Full-attention Q projection emits query plus gate: `q_proj` output is `2 * n_heads * head_dim`. Added `Qwen35FullAttentionShapesFor` to validate these dimensions (`[12288,5120]` for Qwen3.6 27B q_proj, gate size 6144).
- Attention output is multiplied by `sigmoid(gate)` before `o_proj`.
- Linear attention is a gated delta net with conv/recurrent state. This is the largest base-model blocker for Qwen3.6. Added `Qwen35LinearAttentionShapesFor` to make the tensor shape contract explicit before implementing forward/state updates.

### Native MTP graph

`graph_mtp` in `src/models/qwen35.cpp` is the core draft-head algorithm for one native MTP block:

1. Inputs are the next-token id and a pre-norm hidden row `h_p`.
2. Token embedding comes from dedicated MTP embeddings if present, otherwise main `tok_embd`.
3. Normalize both streams separately:

```text
h_norm = RMSNorm(h_input, hnorm)
e_norm = RMSNorm(tok_embd, enorm)
concat = [e_norm ; h_norm]
cur = eh_proj(concat)
```

4. Run one full-attention Qwen3.5 decoder block:

```text
attn_norm
q_proj -> split Q and gate
q_norm, k_proj, k_norm, v_proj
MRoPE on Q/K
GQA attention
attn *= sigmoid(gate)
o_proj
residual
post_attention_layernorm
SwiGLU MLP
residual
```

5. Save pre-output-norm hidden for the next MTP draft step.
6. Apply shared head norm (`mtp.norm` or output norm), then main LM head (or dedicated shared head) for logits.

### Runtime MTP loop

`common/speculative.cpp` has `common_speculative_state_draft_mtp`:

- target and draft contexts both expose `embeddings_pre_norm`;
- `process()` mirrors target verification batches into the MTP context and stores pre-norm hidden rows;
- `pending_h[seq]` carries `(h_p, x_{p+1})` across calls;
- `draft()` feeds `(last token, pending_h)` into the MTP context, samples greedy/top-k, then feeds each drafted token back with the previous MTP pre-norm hidden row to draft multiple tokens;
- `accept(seq, n_accepted)` advances `pending_h` to the hidden row corresponding to the accepted verifier position.

For go-pherence, this maps onto:

- exposing the main CPU/GPU decode pre-output-norm hidden (`h_pre_norm`) as part of `DecodeOne`;
- adding a native MTP draft state with `pending_h`;
- using existing `AcceptMTPDraft` semantics to update `pending_h` and KV cache after verification.

## Shortest implementation path

### Metadata inspection helper

Use `cmd/qwenmtpmeta` for local metadata inspection without entering the full model loader:

```bash
go run ./cmd/qwenmtpmeta -model /path/to/qwen3.6-27b-mtp
# or fail non-zero when native MTP tensors are incomplete:
go run ./cmd/qwenmtpmeta -model /path/to/qwen3.6-27b-mtp -strict
```

It emits JSON with parsed Qwen native-MTP config metadata (including `vocab_size`), whether optional MTP shared-head loading can be attempted, layer counts by type, derived Qwen3.5 full/linear attention shape contracts, any local `mtp.*` safetensors tensor names, optional shared-head tensors as a separate list, summary counts, and `mtp_tensor_complete` when `model.safetensors` or `model.safetensors.index.json` is present. If safetensors are available, it also reports missing required native-MTP tensors for the configured MTP layer count.

### Real-checkpoint smoke runner

Use `cmd/qwen36run` for the current CPU correctness smoke against the downloaded NVFP4 Qwen3.6 checkpoint:

```bash
go run ./cmd/qwen36run -model models/qwen3.6-27b-text-nvfp4-mtp -prompt "Hello" -steps 1 -mtp -mtp-steps 2

# Optional, more expensive prefill diagnostic seeded with the base greedy token:
go run ./cmd/qwen36run -model models/qwen3.6-27b-text-nvfp4-mtp -prompt "Hello" -steps 1 -mtp -greedy-seed

# Sweep newline-separated prompts and summarize MTP acceptance:
go run ./cmd/qwen36run -model models/qwen3.6-27b-text-nvfp4-mtp -sweep prompts.txt -sweep-limit 5 -steps 1 -mtp -mtp-steps 2
```

The runner reports base greedy IDs, native-MTP draft IDs, verifier IDs, acceptance prefix length, and rejection margins. It is intentionally a slow CPU smoke path, not the final public generation API.

### Phase A — metadata and loader recognition

- [x] Add `LlamaConfig` fields for Qwen3.5/Qwen3.6 native MTP metadata:
  - `mtp_num_hidden_layers`
  - `mtp_use_dedicated_embeddings`
  - linear/full attention config fields used by Qwen3.5 text models.
- [x] Add config/tensor-name metadata tests for native `mtp.*` tensor detection without downloading weights.
- [x] Add early clear diagnostic driven by reusable Qwen native-MTP metadata: `qwen3_5_text native MTP detected, base architecture unsupported` instead of failing later on tensor names.

### Phase B — base Qwen3.5/Qwen3.6 model support

Qwen3.6 27B is not plain Qwen3 dense. It uses mixed `linear_attention` / `full_attention` layers and likely `model.language_model.*` nesting.

Needed before MTP can matter:

- [x] load nested `text_config` as `qwen3_5_text` for metadata/early diagnostics;
- [x] map tensor prefix (`model.language_model.` if present) and provide a candidate tensor-source wrapper plus directory-level safetensors base-layer and bundle loaders for loader integration;
- [x] add Qwen3.5/Qwen3.6 base layer structs separate from existing Qwen3 layer assumptions;
- [x] implement full-attention Qwen3.5 layer CPU skeleton:
  - q_proj outputs query + gate;
  - split Q and gate;
  - Q/K RMSNorm;
  - RoPE/MRoPE;
  - GQA attention;
  - multiply attention output by `sigmoid(gate)`;
  - o_proj + residual + post-attention RMSNorm + SwiGLU MLP.
- [x] implement linear-attention/gated-delta-net CPU skeleton (correctness-first scalar recurrence staged; needs real-checkpoint parity):
  - in_proj_qkvz layout and conversion/reorder (projection shape and split primitive staged);
  - conv1d state (state update and depthwise conv primitives staged);
  - beta/alpha/dt/a recurrent update (conservative scalar recurrence staged);
  - z-gated output path;
  - recurrent state cache layout and rollback semantics (forward state exists and clone/checkpoint helpers are staged; speculative integration still pending).
- [ ] support attention output gates if required (`attn_output_gate`, `output_gate_type`);
- [ ] validate BF16/full-precision baseline first, if a non-NVFP4 checkpoint exists.

Linear attention remains the critical parity risk. A focused audit against `am17an/llama.cpp` `qwen35.cpp`/`conversion/qwen.py` fixed several concrete divergences: QKV excludes Z, Z comes from the separate gate projection, conv input is Q/K/V, conv output uses SiLU, convolved Q/K are L2-normalized, beta uses sigmoid, converted A is negative and drives `exp(dt*A)`, and the final linear-attention output uses RMSNorm times SiLU(Z). The remaining unproven part is exact equivalence of the scalar recurrent update and state layout to llama.cpp `build_recurrent_attn`.

### Phase C — native MTP head

Once base forward works:

- [x] define `QwenNativeMTPHead` / layer structs with `fc`, `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `layers`, and `norm`, plus synthetic shape validation, a tensor-source loader contract, single-file/sharded safetensors-backed tensor sources, an auto-selecting safetensors source opener, and a directory-level MTP head loader for BF16/F32 fixtures;
- implement one MTP draft step:
  - normalize embedding and previous hidden separately;
  - concatenate `[embedding || hidden]` to 10240;
  - project through `mtp.fc.weight` to 5120;
  - run the one MTP decoder layer;
  - final `mtp.norm`;
  - use main LM head for logits;
- reuse existing `AcceptMTPDraft` / staged KV commit helpers for verification.

### Phase D — correctness harness

Use `cmd/speccheck` to compare:

- normal greedy generation;
- native-MTP speculative generation with K=1 first;
- then multi-token draft loops.

Do not optimize GPU until CPU token parity is stable.

Synthetic command-line harness while real Qwen3.6 loading is gated:

```bash
go run ./cmd/qwenmtpsynth -steps 2
```

This exercises the native-MTP plan/draft/acceptance/stat plumbing over tiny deterministic tensors and exits non-zero on acceptance failure.

## Immediate execution plan

This supersedes the prior Orthrus/stock-weight speculative exploration as the main implementation track.

1. **Checkpoint triage**
   - find or produce a non-NVFP4 Qwen3.6 27B MTP safetensors artifact if possible;
   - keep inspecting NVFP4 metadata so the current public checkpoint remains useful as a shape/layout oracle;
   - do not enable public NVFP4 generation until real logits/tokens agree.
2. **Loader/config gate**
   - add metadata/header tests and explicit unsupported diagnostics for `qwen3_5_text + mtp_num_hidden_layers`;
   - avoid silent partial loads or generic missing-tensor errors.
3. **Base Qwen3.6 text model**
   - implement nested `text_config` loading and tensor prefix mapping;
   - implement or explicitly stage support for `linear_attention` / `full_attention` layers;
   - validate full-attention-only synthetic fixtures before touching large weights.
4. **Native MTP head**
   - load `mtp.fc`, pre-FC norms, one MTP decoder layer, and `mtp.norm`;
   - implement K=1/multi-step greedy draft step (CPU skeleton now covers preprojection plus full-attention/MLP MTP block, RoPE when frequency tables are provided, current K/V return, history-aware attention over past+current MTP KV, final MTP norm + main LM-head logits/argmax, `QwenNativeMTPDraftState`, bounded `DraftSteps`, `QwenNativeMTPPlan`, plan-based adapters to existing `AcceptMTPDraft` from verifier tokens or validated logits, accepted-prefix draft-state commit, and native-MTP stats with aggregation helpers; generation-loop integration remains next);
   - reuse `AcceptMTPDraft` and `speccheck` parity checks.
5. **Correctness harness**
   - extend `speccheck` with a native-MTP mode once the draft step exists (flag placeholder `-qwen-native-mtp` now fails clearly until real `LoadLlama` integration lands; synthetic native-MTP correctness is covered in model tests);
   - store golden token baselines for small prompts;
   - only optimize after CPU parity is stable.
6. **Performance path**
   - replace replay verification with KV-reusing verifier block;
   - then move hot MTP and verifier paths to NVIDIA.
