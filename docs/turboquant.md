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
- `cmd/llmserver /health` reports TurboQuant and REAP together when both are
  active, so deployments can verify cache policy interpretation, KV/protected
  layer accounting, byte estimates, and REAP source without starting a generation
  request.

## Validation snapshot

The local Qwen3.6 REAP GGUF validation bundle is:

```bash
GOTMPDIR=$PWD/.gotmp make gguf-inspect-qwen36-reap
GOTMPDIR=$PWD/.gotmp make gguf-smoke-qwen36-reap
GOTMPDIR=$PWD/.gotmp make gguf-validate-qwen36-reap
GOTMPDIR=$PWD/.gotmp make gguf-bench-qwen36-reap
GOTMPDIR=$PWD/.gotmp make gguf-check-qwen36-reap  # validation + benchmark
GOTMPDIR=$PWD/.gotmp make gguf-ci-qwen36-reap     # focused build smoke + check
```

Current expected checkpoint inventory:

- architecture: `qwen35moe`
- name marker: `REAP20`
- REAP ratio/source: `0.20` / `filename_or_name`
- tensors: `733` (`F32=301`, `Q4_K=371`, `Q6_K=61`)
- layers/hidden/heads/vocab/tokenizer/context: `40` / `2048` / `16` / `248320` / `248320` / `262144`
- tokenizer stops: BOS `248044`, EOS `248046`
- MoE: `205` experts, `8` active per token
- KV: `kv_dim=512`, `cache_layers=10`, `protected_cache_layers=1`
- synthetic KV append smoke: layer `3`, compressed/full `3`/`2`, bytes `9440`
- one-token runtime KV plan: `245760` F32 bytes and `81920` compressed bytes
- one-token greedy smoke/bench: `prompt_ids=0 -> generated=[489]`, decoded as `"ype"`; bench KV bytes are `245760` F32 and `81920` compressed

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


## Native SIMD rotation path

TurboQuant vector rotation and inverse rotation now call the checked `backends/simd/runtime` GEMV facade. On supported CPUs this routes the hot per-head rotation dot products through AVX2/FMA, NEON, or RVV-gated dot-product assembly where available, while retaining a scalar fallback for malformed inputs or unsupported runtimes. Built-in key/value rotations store transposed matrices so inverse rotation also uses row-GEMV/dot-product dispatch rather than strided scalar column walks. This keeps GGUF REAP/TurboQuant execution inside go-pherence pure Go/SIMD components instead of relying on llama.cpp cache kernels.


`GGUF_EXPECT_SIMD_ROTATION=1` can be passed to `make gguf-inspect`/`make gguf-validate` to require that the host reports native SIMD dot-product support for TurboQuant rotation in both the inspect-time plan and runtime smoke paths.


## KV-owned key/value helpers

`TurboQuantState` now exposes key/value-specific quantize/dequantize helpers for compressed KV cache users. The compressed cache calls these helpers instead of passing raw rotation/codebook slices around, so built-in K/V cache operations consistently use the owned TurboQuant policy and the stored-transpose SIMD inverse path. The generic vector API remains available for tests and compatibility.


## Scratch-owned dequantization

`CompressedKVCache.GetK/GetV` now use destination-based key/value dequantize helpers and write decompressed heads directly into reusable scratch buffers. This removes the previous per-compressed-head restored-vector allocation while preserving malformed-entry fallback behavior.


## Model-side SIMD readiness

Model-side `GGUFTurboQuantPlan` now carries the kv-owned SIMD readiness fields (`simd_arch`, `simd_rotation`, `simd_vec`, `simd_avx2`, `simd_neon`, `simd_rvv`). Runtime tools such as `ggufsmoke` can report the model plan directly instead of reinterpreting SIMD capabilities at the CLI layer.


## Generation runtime-plan SIMD readiness

Generation KV runtime plans now carry the same SIMD readiness fields as the static TurboQuant plan. `ggufsmoke` reports these fields on `turboquant_runtime_kv`, so prompt/max-new allocation diagnostics show both cache byte accounting and native SIMD rotation readiness from the model runtime plan.


## Scratch-owned quantization

`CompressedKVCache` compression now uses scratch-aware K/V quantize helpers, reusing per-cache rotated-coordinate and index buffers while preserving the public allocating `QuantizeKey`/`QuantizeValue` compatibility methods. This reduces per-head allocation pressure in the native TurboQuant cache append path.


## Destination-packed quantization

`CompressedKVCache` compression now preallocates exact packed K/V entry storage and writes each head directly into its destination slice via destination-packed quantize helpers. This removes the previous temporary packed-slice allocation per compressed head while retaining the public allocating compatibility API.


## Scratch-owned unpack/dequantization

`CompressedKVCache` decompression now uses scratch-aware unpack/dequantize helpers, reusing per-cache unpacked-index and rotated-coordinate buffers while writing final K/V heads into the sequence scratch. This removes the remaining per-head temporary allocation in the native TurboQuant cache read path.


## Stored vs scratch byte accounting

TurboQuant cache accounting now separates stored payload bytes from reusable scratch bytes. `MemoryBytes()` remains the stored full/compressed K/V payload estimate used by existing expectations, while `ScratchBytes()` and `TotalMemoryBytes()` expose reusable SIMD quant/dequant scratch. `ggufsmoke` cache smoke prints `stored_bytes`, `scratch_bytes`, and `total_bytes` alongside the legacy `bytes` field.


`GGUF_EXPECT_KV_SMOKE_SCRATCH_BYTES` and `GGUF_EXPECT_KV_SMOKE_TOTAL_BYTES` can be used with `make gguf-turboquant-smoke`/`make gguf-validate` to assert reusable TurboQuant scratch and total cache footprint alongside the legacy stored-byte assertion.


### Qwen3.6 REAP cache-smoke scratch assertion values

For the local `/opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf` preset with `turbo4/turbo2`, residual window `2`, and `-kv-smoke-tokens 5`, the pinned native TurboQuant cache-smoke byte values are:

```text
stored_bytes=9440
scratch_bytes=1280
total_bytes=10720
```

The `gguf-validate-qwen36-reap` target asserts all three values.


## KV-owned cache stats

`CompressedKVCache.Stats()` now owns cache-smoke accounting for sequence length, compressed/full counts, stored bytes, scratch bytes, and total bytes. Tools such as `ggufsmoke` consume this kv-owned summary instead of manually combining cache counters and byte methods.


## Runtime scratch estimates

Generation runtime plans now estimate reusable TurboQuant scratch bytes before cache materialization. `runtime/kv.EstimateTurboQuantScratchBytes` accounts for per-layer quant/dequant rotated/index scratch plus GetK/GetV sequence scratch for unprotected compressed layers. `ggufsmoke` reports `estimated_scratch_bytes` and `estimated_total_bytes` on `turboquant_runtime_kv`.


`GGUF_EXPECT_RUNTIME_SCRATCH_BYTES` and `GGUF_EXPECT_RUNTIME_TOTAL_BYTES` can be used with `make gguf-smoke`/`make gguf-validate` to assert generation runtime-plan scratch and total KV+scratch byte estimates.


### Qwen3.6 REAP runtime-plan scratch assertion values

For the local `/opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf` preset with prompt ID `0`, `max-new=1`, `turbo4/turbo2`, and residual window `2`, the pinned generation runtime-plan values are:

```text
float_alloc_bytes=245760
compressed_estimated_bytes=81920
estimated_scratch_bytes=96768
estimated_total_bytes=424448
```

The `gguf-validate-qwen36-reap` target asserts all four runtime-plan byte values.


`GGUF_EXPECT_KV_SCRATCH_BYTES` and `GGUF_EXPECT_KV_TOTAL_BYTES` can be used with `make gguf-bench` to assert post-generation compressed-cache scratch and total KV+scratch bytes alongside stored float/compressed byte counters.


### Qwen3.6 REAP benchmark KV footprint values

For the local one-token Qwen3.6 REAP benchmark (`prompt-ids=0`, `max-new=1`, `turbo4/turbo2`, residual window `2`), the pinned post-generation KV footprint is:

```text
kv_float_bytes=245760
kv_compressed_bytes=81920
kv_scratch_bytes=0
kv_total_bytes=327680
```

The one-token benchmark does not materialize compressed-cache read scratch, so `kv_scratch_bytes=0`; runtime-plan estimates still report the scratch that would be needed when compressed cache reads are materialized.
