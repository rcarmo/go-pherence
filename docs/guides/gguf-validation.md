# GGUF and TurboQuant validation

[Command index](commands.md) | [Runtime tuning](tuning.md)

## GGUF REAP/TurboQuant smoke

The native GGUF path is pure Go/SIMD and does not shell out to llama.cpp. It accepts llama.cpp-compatible cache policy names and maps them to `runtime/kv` TurboQuant caches:

```bash
make gguf-inspect \
  GGUF_MODEL=/opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf \
  GGUF_CACHE_TYPE_K=turbo4 \
  GGUF_CACHE_TYPE_V=turbo2 \
  GGUF_KV_RESIDUAL_WINDOW=128

make gguf-bench \
  GGUF_MODEL=/opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf \
  GGUF_PROMPT_IDS=0 \
  GGUF_MAX_NEW=1 \
  GGUF_EXPECT_GENERATED=489 \
  GGUF_CACHE_TYPE_K=turbo4 \
  GGUF_CACHE_TYPE_V=turbo2 \
  GGUF_KV_RESIDUAL_WINDOW=2

# Combined inspect + generation assertion + synthetic compressed-KV append smoke
make gguf-validate \
  GGUF_MODEL=/opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf \
  GGUF_EXPECT_GENERATED=489 \
  GGUF_EXPECT_DECODED=ype \
  GGUF_EXPECT_REAP_RATIO=0.20 \
  GGUF_EXPECT_REAP_SOURCE=filename_or_name \
  GGUF_EXPECT_ARCHITECTURE=qwen35moe \
  GGUF_EXPECT_NAME_CONTAINS=REAP20 \
  GGUF_EXPECT_TENSOR_COUNT=733 \
  GGUF_EXPECT_LAYERS=40 \
  GGUF_EXPECT_HIDDEN_SIZE=2048 \
  GGUF_EXPECT_HEADS=16 \
  GGUF_EXPECT_VOCAB_SIZE=248320 \
  GGUF_EXPECT_TOKENIZER_TOKENS=248320 \
  GGUF_EXPECT_BOS=248044 \
  GGUF_EXPECT_EOS=248046 \
  GGUF_EXPECT_MAX_SEQ_LEN=262144 \
  GGUF_EXPECT_FULL_ATTENTION_INTERVAL=4 \
  GGUF_EXPECT_KV_HEADS=2 \
  GGUF_EXPECT_HEAD_DIM=256 \
  GGUF_EXPECT_KV_DIM=512 \
  GGUF_EXPECT_EXPERTS=205 \
  GGUF_EXPECT_EXPERTS_PER_TOKEN=8 \
  GGUF_EXPECT_F32_COUNT=301 \
  GGUF_EXPECT_Q4_K_COUNT=371 \
  GGUF_EXPECT_Q6_K_COUNT=61 \
  GGUF_EXPECT_CACHE_LAYERS=10 \
  GGUF_EXPECT_PROTECTED_CACHE_LAYERS=1 \
  GGUF_EXPECT_FULL_KV_BYTES=10737418240 \
  GGUF_EXPECT_ESTIMATED_KV_BYTES=2055275200 \
  GGUF_EXPECT_SAVED_KV_BYTES=8682143040 \
  GGUF_EXPECT_KV_SMOKE_LAYER=3 \
  GGUF_EXPECT_KV_SMOKE_COMPRESSED=3 \
  GGUF_EXPECT_KV_SMOKE_FULL=2 \
  GGUF_EXPECT_KV_SMOKE_BYTES=9440 \
  GGUF_KV_RESIDUAL_WINDOW=2

# Generic inspect/smoke/cache-smoke plus benchmark
make gguf-check \
  GGUF_MODEL=/opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf \
  GGUF_EXPECT_GENERATED=489 \
  GGUF_EXPECT_DECODED=ype \
  GGUF_EXPECT_RUNTIME_FLOAT_BYTES=245760 \
  GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES=81920 \
  GGUF_EXPECT_KV_FLOAT_BYTES=245760 \
  GGUF_EXPECT_KV_COMPRESSED_BYTES=81920 \
  GGUF_EXPECT_REAP_RATIO=0.20 \
  GGUF_EXPECT_REAP_SOURCE=filename_or_name \
  GGUF_EXPECT_ARCHITECTURE=qwen35moe \
  GGUF_EXPECT_NAME_CONTAINS=REAP20 \
  GGUF_EXPECT_TENSOR_COUNT=733 \
  GGUF_EXPECT_LAYERS=40 \
  GGUF_EXPECT_HIDDEN_SIZE=2048 \
  GGUF_EXPECT_HEADS=16 \
  GGUF_EXPECT_VOCAB_SIZE=248320 \
  GGUF_EXPECT_TOKENIZER_TOKENS=248320 \
  GGUF_EXPECT_BOS=248044 \
  GGUF_EXPECT_EOS=248046 \
  GGUF_EXPECT_MAX_SEQ_LEN=262144 \
  GGUF_EXPECT_FULL_ATTENTION_INTERVAL=4 \
  GGUF_EXPECT_KV_HEADS=2 \
  GGUF_EXPECT_HEAD_DIM=256 \
  GGUF_EXPECT_KV_DIM=512 \
  GGUF_EXPECT_EXPERTS=205 \
  GGUF_EXPECT_EXPERTS_PER_TOKEN=8 \
  GGUF_EXPECT_F32_COUNT=301 \
  GGUF_EXPECT_Q4_K_COUNT=371 \
  GGUF_EXPECT_Q6_K_COUNT=61 \
  GGUF_EXPECT_CACHE_LAYERS=10 \
  GGUF_EXPECT_PROTECTED_CACHE_LAYERS=1 \
  GGUF_EXPECT_FULL_KV_BYTES=10737418240 \
  GGUF_EXPECT_ESTIMATED_KV_BYTES=2055275200 \
  GGUF_EXPECT_SAVED_KV_BYTES=8682143040 \
  GGUF_EXPECT_KV_SMOKE_LAYER=3 \
  GGUF_EXPECT_KV_SMOKE_COMPRESSED=3 \
  GGUF_EXPECT_KV_SMOKE_FULL=2 \
  GGUF_EXPECT_KV_SMOKE_BYTES=9440 \
  GGUF_KV_RESIDUAL_WINDOW=2

# Generic focused package smoke + inspect/smoke/cache-smoke validation
make gguf-ci \
  GGUF_MODEL=/opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf \
  GGUF_EXPECT_GENERATED=489 \
  GGUF_EXPECT_DECODED=ype \
  GGUF_EXPECT_REAP_RATIO=0.20 \
  GGUF_EXPECT_REAP_SOURCE=filename_or_name \
  GGUF_EXPECT_ARCHITECTURE=qwen35moe \
  GGUF_EXPECT_NAME_CONTAINS=REAP20 \
  GGUF_EXPECT_TENSOR_COUNT=733 \
  GGUF_EXPECT_LAYERS=40 \
  GGUF_EXPECT_HIDDEN_SIZE=2048 \
  GGUF_EXPECT_HEADS=16 \
  GGUF_EXPECT_VOCAB_SIZE=248320 \
  GGUF_EXPECT_TOKENIZER_TOKENS=248320 \
  GGUF_EXPECT_BOS=248044 \
  GGUF_EXPECT_EOS=248046 \
  GGUF_EXPECT_MAX_SEQ_LEN=262144 \
  GGUF_EXPECT_FULL_ATTENTION_INTERVAL=4 \
  GGUF_EXPECT_KV_HEADS=2 \
  GGUF_EXPECT_HEAD_DIM=256 \
  GGUF_EXPECT_KV_DIM=512 \
  GGUF_EXPECT_EXPERTS=205 \
  GGUF_EXPECT_EXPERTS_PER_TOKEN=8 \
  GGUF_EXPECT_F32_COUNT=301 \
  GGUF_EXPECT_Q4_K_COUNT=371 \
  GGUF_EXPECT_Q6_K_COUNT=61 \
  GGUF_EXPECT_CACHE_LAYERS=10 \
  GGUF_EXPECT_PROTECTED_CACHE_LAYERS=1 \
  GGUF_EXPECT_FULL_KV_BYTES=10737418240 \
  GGUF_EXPECT_ESTIMATED_KV_BYTES=2055275200 \
  GGUF_EXPECT_SAVED_KV_BYTES=8682143040 \
  GGUF_EXPECT_KV_SMOKE_LAYER=3 \
  GGUF_EXPECT_KV_SMOKE_COMPRESSED=3 \
  GGUF_EXPECT_KV_SMOKE_FULL=2 \
  GGUF_EXPECT_KV_SMOKE_BYTES=9440 \
  GGUF_KV_RESIDUAL_WINDOW=2

# Convenience targets for the local Qwen3.6 REAP checkpoint/expectations above
make gguf-inspect-qwen36-reap
make gguf-smoke-qwen36-reap
make gguf-validate-qwen36-reap
make gguf-bench-qwen36-reap
make gguf-check-qwen36-reap
make gguf-ci-qwen36-reap
```

`gguf-check` is the generic validation-plus-benchmark target. `gguf-ci` is the generic focused package smoke plus `gguf-validate`; `gguf-ci-qwen36-reap` runs the focused build-only package smoke (`GGUF_CI_PACKAGES`, overrideable) before `gguf-check-qwen36-reap`; `llmserver /health` exposes the same TurboQuant byte estimate plus KV/protected-layer accounting for server deployments. `gguf-check-qwen36-reap` runs the validation target plus the benchmark target, including expected runtime and benchmark KV byte assertions (`245760` F32 bytes and `81920` compressed bytes for the one-token local smoke). `ggufinspect` reports REAP ratio/source, runtime readiness, KV dimensions, cache-layer counts, protected-layer counts, and full-vs-compressed KV byte estimates. `ggufsmoke -bench` runs the same generation allocator used by `GenerateWithOptions` and reports prefill/decode timing plus actual F32/compressed KV bytes for the run; `make gguf-bench-qwen36-reap` bundles the local expected token/decoded-text assertions with that benchmark. Set `GGUF_EXPECT_REAP_RATIO`, `GGUF_EXPECT_REAP_SOURCE`, `GGUF_EXPECT_ARCHITECTURE`, `GGUF_EXPECT_NAME_CONTAINS`, `GGUF_EXPECT_TENSOR_COUNT`, `GGUF_EXPECT_LAYERS`, `GGUF_EXPECT_HIDDEN_SIZE`, `GGUF_EXPECT_HEADS`, `GGUF_EXPECT_VOCAB_SIZE`, `GGUF_EXPECT_TOKENIZER_TOKENS`, `GGUF_EXPECT_BOS`, `GGUF_EXPECT_EOS`, `GGUF_EXPECT_MAX_SEQ_LEN`, `GGUF_EXPECT_FULL_ATTENTION_INTERVAL`, `GGUF_EXPECT_KV_HEADS`, `GGUF_EXPECT_HEAD_DIM`, `GGUF_EXPECT_KV_DIM`, `GGUF_EXPECT_EXPERTS`, `GGUF_EXPECT_EXPERTS_PER_TOKEN`, `GGUF_EXPECT_F32_COUNT`, `GGUF_EXPECT_Q4_K_COUNT`, `GGUF_EXPECT_Q6_K_COUNT`, `GGUF_EXPECT_CACHE_LAYERS`, `GGUF_EXPECT_PROTECTED_CACHE_LAYERS`, `GGUF_EXPECT_FULL_KV_BYTES`, `GGUF_EXPECT_ESTIMATED_KV_BYTES`, `GGUF_EXPECT_SAVED_KV_BYTES` (or the matching `ggufinspect` flags) and `GGUF_EXPECT_GENERATED`/`GGUF_EXPECT_DECODED` (or `ggufsmoke -expect-generated`/`-expect-decoded`) to make validation fail if REAP metadata/source inference, runtime shape/MoE planning, TurboQuant layer planning, synthetic compressed-KV smoke accounting, or greedy output/decoded text changes unexpectedly.


### TurboQuant SIMD readiness in server health

`cmd/llm/llmserver /health` includes native SIMD dispatch diagnostics inside the `turboquant` object when cache policy flags are set:

```json
{
  "turboquant": {
    "simd_arch": "amd64",
    "simd_rotation": true,
    "simd_vec": true,
    "simd_avx2": true,
    "simd_neon": false,
    "simd_rvv": false
  }
}
```

`simd_rotation=true` means TurboQuant per-head rotation can use the checked go-pherence SIMD dot-product facade for the active CPU (AVX2/FMA, NEON, or RVV where available), with scalar fallback otherwise.


`GGUF_EXPECT_SIMD_ROTATION=1` can be passed to `make gguf-inspect`/`make gguf-validate` to require that the host reports native SIMD dot-product support for TurboQuant rotation in both the inspect-time plan and runtime smoke paths.


`GGUF_EXPECT_KV_SMOKE_SCRATCH_BYTES` and `GGUF_EXPECT_KV_SMOKE_TOTAL_BYTES` can be used with `make gguf-turboquant-smoke`/`make gguf-validate` to assert reusable TurboQuant scratch and total cache footprint alongside the legacy stored-byte assertion.


### Qwen3.6 REAP cache-smoke scratch assertion values

For the local `/opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf` preset with `turbo4/turbo2`, residual window `2`, and `-kv-smoke-tokens 5`, the pinned native TurboQuant cache-smoke byte values are:

```text
stored_bytes=9440
scratch_bytes=1280
total_bytes=10720
```

The `gguf-validate-qwen36-reap` target asserts all three values.


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


### Qwen3.6 REAP benchmark aggregate KV counters

`ggufsmoke -bench` now reports kv-owned aggregate compressed-cache counters in addition to byte totals. For the local one-token Qwen3.6 REAP benchmark, the pinned observation is:

```text
kv_compressed_layers=10
kv_seq=2
kv_compressed_count=0
kv_full_count=20
kv_float_bytes=245760
kv_compressed_bytes=81920
kv_scratch_bytes=0
kv_total_bytes=327680
```


`GGUF_EXPECT_KV_COMPRESSED_LAYERS`, `GGUF_EXPECT_KV_SEQ`, `GGUF_EXPECT_KV_COMPRESSED_COUNT`, and `GGUF_EXPECT_KV_FULL_COUNT` can be used with `make gguf-bench` to assert aggregate compressed-cache counters from `runtime/kv.AggregateCompressedKVCacheStats`.


### Qwen3.6 REAP preset requires SIMD rotation readiness

The local Qwen3.6 REAP validation/benchmark presets now set `GGUF_EXPECT_SIMD_ROTATION=1`, so `make gguf-ci-qwen36-reap` fails if the host cannot report native SIMD dot-product support for TurboQuant rotation through the go-pherence SIMD facade.


`GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES` and `GGUF_EXPECT_ESTIMATED_TOTAL_BYTES` can be used with `make gguf-inspect`/`make gguf-validate` to assert inspect-time TurboQuant KV+scratch estimates. The Qwen3.6 REAP preset pins `estimated_scratch_bytes=9663699456` and `estimated_total_bytes=11718974656` for the full-context `turbo4/turbo2` plan.


`llmserver /health` now reports TurboQuant `estimated_scratch_bytes` and `estimated_total_bytes` alongside stored KV estimates, using the same `runtime/kv` estimator as inspect/runtime tooling.


`ggufsmoke` also accepts `-expect-estimated-scratch-bytes` and `-expect-estimated-total-bytes`, so smoke/cache-smoke/bench paths can assert the same static/full-context TurboQuant plan values as `ggufinspect`.


`ggufsmoke` static plan assertions also cover full/estimated/saved KV bytes (`-expect-full-kv-bytes`, `-expect-estimated-kv-bytes`, `-expect-saved-kv-bytes`) in addition to scratch and total estimates, so smoke paths can pin the complete static TurboQuant plan.
