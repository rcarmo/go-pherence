# TurboQuant KV Cache Compression

TurboQuant lives in `runtime/kv` and is an optional CPU-backend KV cache compression path for long-context
inference.

## Usage

```bash
# CPU backend with compressed KV cache
./bin/llmgen -model models/gemma4-e2b-mlx4 -prompt "..." -tokens 256 --turbo-quant

# Existing environment variable remains supported for diagnostics
TURBO_QUANT=1 ./bin/llmgen -model models/gemma4-e2b-mlx4 -prompt "..." -tokens 256
```

`--turbo-quant` currently applies to the CPU backend only. If `--gpu` is also
provided, the CLI prints a warning; GPU KV-cache compression is not wired yet.

For GGUF/llama.cpp-compatible REAP checkpoints, use the native GGUF smoke and
inspection commands. The llama.cpp-style policy names are mapped to the same
`runtime/kv` implementation; no external llama.cpp cache runtime is used:

```bash
# Lightweight metadata/readiness + native TurboQuant byte plan
make gguf-inspect \
  GGUF_MODEL=/opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf \
  GGUF_CACHE_TYPE_K=turbo4 \
  GGUF_CACHE_TYPE_V=turbo2 \
  GGUF_KV_RESIDUAL_WINDOW=128

# One-token generation/benchmark through the pure Go/SIMD GGUF path
make gguf-bench \
  GGUF_MODEL=/opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf \
  GGUF_PROMPT_IDS=0 \
  GGUF_MAX_NEW=1 \
  GGUF_CACHE_TYPE_K=turbo4 \
  GGUF_CACHE_TYPE_V=turbo2 \
  GGUF_KV_RESIDUAL_WINDOW=2
```

## Current implementation

- `runtime/kv` owns `TurboQuantState`, `CompressedKVCache`, and generic float/compressed KV checkpoint/rollback helpers.
- The `model` package owns model-specific KV dimension derivation and CPU generation wiring.
- Per-layer `CompressedKVCache` wrapper for CPU `LlamaModel.Generate`.
- GGUF `GGUFLlama.GenerateWithOptions` accepts llama.cpp-compatible `cache_type_k`/`cache_type_v` names and attaches native compressed caches to the layers that actually use autoregressive KV. Plain LLaMA-family GGUF models use every layer; QwenNext/REAP hybrid GGUF models use only full-attention interval layers while recurrent/SSM layers keep their own state.
- Recent tokens stay full precision via a 128-token residual window.
- `CompressedKVCache` constructor sizing, accessors, scratch-buffer sizing, packed-entry validation, and memory accounting use checked/saturating arithmetic so malformed dimensions fail closed.
- Older tokens are compressed on append.
- First/last protected layers stay full precision.
- K uses 4-bit quantization; V uses 2-bit quantization.
- K/V are rotated by deterministic random orthogonal matrices before quantizing.
- Gemma4 variable head dimensions are handled by caching one TurboQuant state per
  `headDim`.
- Attention reads call `GetK()`/`GetV()`, which decompress into reusable scratch
  buffers to avoid per-token full-cache allocation churn.
- Staged checkpoint/restore/keep-prefix helpers support speculative verifier
  rollback and accepted-prefix commit even when candidate appends cross the
  residual window and trigger compression.
- `runtime/kv.EstimateTurboQuantKV` is the shared byte estimator for model plans,
  `ggufinspect`, `ggufsmoke`, and `cmd/llmserver /health`; it reports full bytes,
  estimated compressed bytes, savings, ratio, KV layer count, and protected-layer
  count.

## Validation snapshot

The local Qwen3.6 REAP GGUF validation bundle is:

```bash
GOTMPDIR=$PWD/.gotmp make gguf-smoke-qwen36-reap
GOTMPDIR=$PWD/.gotmp make gguf-validate-qwen36-reap
GOTMPDIR=$PWD/.gotmp make gguf-bench-qwen36-reap
```

Current expected checkpoint inventory:

- architecture: `qwen35moe`
- name marker: `REAP20`
- REAP ratio/source: `0.20` / `filename_or_name`
- tensors: `733` (`F32=301`, `Q4_K=371`, `Q6_K=61`)
- layers/vocab/context: `40` / `248320` / `262144`
- tokenizer stops: BOS `248044`, EOS `248046`
- MoE: `205` experts, `8` active per token
- KV: `kv_dim=512`, `cache_layers=10`, `protected_cache_layers=1`
- one-token greedy smoke/bench: `prompt_ids=0 -> generated=[489]`, decoded as `"ype"`

Unit tests cover:

- Roundtrip quality: 4-bit K max error about `0.455`, 2-bit V max error about
  `1.715` for the synthetic test vector.
- Compression layout: 200-token cache with a 128-token residual window stores 72
  compressed and 128 full-precision entries.
- Protected layers never compress. The protected-layer helper is nil-safe and rejects negative query indices before applying configured negative aliases such as `-1`/`-2` for last layers.
- Short prompts under the residual window produce the same first tokens as the
  uncompressed CPU path.

## Limitations

- CPU path only.
- Quantization is currently range-adaptive uniform after rotation. The code keeps
  analytic codebook scaffolding, but non-uniform/Beta-optimal levels are not used
  yet because the current uniform path has lower measured error in tests.
- Decompression still happens before the existing attention kernel; a future
  optimized path should read compressed blocks directly or cache per-window
  dequantized pages.
- GPU-side compression/decompression and compressed KV attention are future work.
- MTP/speculative decoding can commit/rollback TurboQuant-backed verifier KV via
  the staging helpers, but the current internal verifier-forward loop itself is
  float-KV CPU-only and public speculative generation remains disabled.
