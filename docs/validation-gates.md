# Validation gates

This repository uses phase-level validation for backend coverage work. Do not run the full test/vet matrix after every small mechanical change; run it when a complete plan phase is ready to validate.

## Standard phase gate

Use the workspace-local Go temp directory in constrained containers:

```bash
GOTMPDIR=$PWD/.gotmp go test ./...
GOTMPDIR=$PWD/.gotmp go vet ./...
make test-cpu
```

`make test-cpu` keeps NVIDIA disabled and Vulkan software devices opt-in only, so CPU/scalar fallback behavior is checked without requiring GPU hardware.

## Optional phase-specific gates

Run these only when the corresponding phase is being accepted:

```bash
# SIMD/CPU backend phases
GOTMPDIR=$PWD/.gotmp go test ./backends/simd/... ./backends/mlx ./model

# NVIDIA runtime phase on CPU-only hosts
GOTMPDIR=$PWD/.gotmp go test ./backends/nvidia/runtime ./backends/nvidia/ioctl

# Vulkan wrapper phase without a Vulkan device
GOTMPDIR=$PWD/.gotmp go test ./backends/vulkan

# Whisper large-v3-turbo execution graph (CPU/SIMD transcript parity, audio commands, mel, SIMD, prompt/timestamp/speculative scaffolds).
# Treat the native CPU/SIMD path as the oracle for NEON, RISC-V/IME, A100,
# NVIDIA, and CUDA/PTX refinements; hardware paths must preserve these results.
# The transcript parity target fails loudly if local turbo weights/tokenizer/JFK audio are missing.
# The SIMD parity target is a required scalar-oracle gate for locally runnable
# Whisper vectorized paths (mel/filterbank, conv1d, attention/GEMV, dot kernels).
# The CUDA parity target is a numeric CPU-oracle gate for the opt-in CUDA graph
# surfaces; it may skip on CPU-only hosts, but on CUDA hosts it must initialize
# the NVIDIA runtime, load the Whisper PTX entries, and run real numeric assertions
# instead of silently falling back. Current shared-host evidence: RTX 3060 loads
# all 78 mega-module kernels and TestGPUEncoderForward passes on large-v3-turbo
# with max_diff≈1.9e-4.
# The GPU graph parity target runs the same JFK transcript contract with the
# umbrella GPU graph flag enabled, exercising real CUDA dispatch on CUDA hosts
# while preserving CPU/SIMD fallback behavior on CPU-only hosts.
make whisper-turbo-parity
make whisper-simd-parity
make whisper-cuda-parity
make whisper-gpu-graph-parity
make whisper-turbo-check

# Optional, non-gating GPU timing smoke. This uses the current turbo assets and
# is intentionally opt-in because host load can dominate RTF measurements.
WHISPER_RUN_GPU_RTF=1 GOTMPDIR=$PWD/.gotmp go test ./models/whisper \
  -run TestGPURTFEstimate -count=1 -v

# Equivalent explicit form:
GOTMPDIR=$PWD/.gotmp go test ./models/whisper ./loader/audio ./backends/simd/fft ./backends/simd/runtime \
  -run 'TestWhisperConv1DFastMatchesScalarOracle|TestWhisperLayerNormUsesSIMDOracleMatchesScalar|TestWhisperFullAttentionMatchesScalarOracle|TestLinearRowBlockUsesSIMDOracleMatchesScalar|TestMelSpectrogramMatchesReferencePath|TestMelSpectrogramFusedUsesLog10|TestDotI8F32|TestDotI8F32x4|TestSdotx4|TestQ4RowDot' \
  -count=1 -v
GOTMPDIR=$PWD/.gotmp go test ./models/whisper \
  -run 'TestGPUEncoderForwardNotReadyFallbackMatchesCPU|TestWhisperCUDA|TestWhisperGPUGraphUmbrella|TestWhisperGPUFeatureFlags|TestNewDecoderStateGPU' \
  -count=1 -v
GOTMPDIR=$PWD/.gotmp go test ./backends/nvidia/runtime \
  -run TestWhisperAttentivePoolParity -count=1 -v
WHISPER_REQUIRE_TURBO_PARITY=1 GO_PHERENCE_WHISPER_GPU_GRAPH=1 \
  GOTMPDIR=$PWD/.gotmp go test ./models/whisper \
  -run TestLargeV3TurboJFKCPUTranscriptParity -count=1 -v
GOTMPDIR=$PWD/.gotmp go test \
  ./models/whisper \
  ./cmd/audio/... \
  ./loader/audio \
  ./backends/simd/fft \
  ./backends/simd/runtime \
  ./backends/cuda/ptx
python3 scripts/whisper_turbo_smoke.py --audio testdata/jfk.wav
# whisper_turbo_smoke covers standalone translate/transcribe, standalone chunked
# no-timestamp, standalone timestamp VTT for translate/transcribe, standalone timestamp+diarize VTT,
# diarize-vtt translate/transcribe, and diarize-vtt+speaker for both translate
# and transcribe.
python3 scripts/speakercheck_suite.py testdata/speakercheck_suite.json

# Optional backend parity checks against the native SIMD oracle. A100 comparisons
# cover plain stdout decode for translate/transcribe, standalone timestamp/VTT
# for translate/transcribe, diarize-vtt output, and diarize-vtt with speaker
# labels for translate/transcribe. K3/RISC-V int8 checks compare stdout,
# standalone timestamp/VTT, diarize-vtt VTT, and speaker-tagged diarize-vtt VTT
# for translate/transcribe;
# on non-riscv hosts, stubs keep this a command/prompt smoke.
make whisper-backend-compare
make whisper-backend-podcast-compare
# podcast target compares stdout, standalone timestamp/VTT, diarize-vtt VTT, and speaker-tagged diarize-vtt VTT on a 12s long-form window for A100 and int8 backends.
# Pass repeated --audio flags and optional --start/--duration to
# scripts/whisper_a100_compare.py for broader/long-form clip sets.

# DiffusionGemma runnable llama.cpp GGUF golden/probe gate. The first part is
# fixture-only and does not require local 48 GiB shards or CUDA; it locks prompt
# IDs, reference response IDs, current Go-vs-llama mismatch metadata, first-step
# top-logit probe values, phase-aligned row28 structural trace parity
# (ffn_post_norm/l_out across all text layers), row28 input-norm and layer-op
# parity gates, plus Q4_K/Q8_0/Q5_0/Q6_K direct quant oracles. The second part
# runs the local Q4_K_M GGUF tiny forward golden when that GGUF is present,
# isolated in its own go test process because it owns mmap-backed GGUF buffers.
make diffusiongemma-golden-gate

# Diagnostic package import/build boundary, when local diagnostic assets permit it
GOTMPDIR=$PWD/.gotmp go test -tags diagnostic ./model/gemma4

# Gemma4 31B packed MTP smoke, when local ignored model assets are present
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -model models/gemma4-31b-it-4bit \
  -mtp-drafter models/gemma4-31b-it-mtp-assistant-4bit \
  -mtp-smoke \
  -prompt "Hello"

# Recommended real-prompt Gemma4 E4B QAT GGUF + BF16 MTP smoke on the local RTX 3060 profile
# If the verifier/drafter snapshots are missing, fetch/provision the local GGUF pair first.
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -gpu -gpu-layers 0 \
  -model models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf \
  -mtp-drafter models/gemma4-e4b-it-google-qat-gguf/MTP/gemma-4-E4B-it-BF16-MTP.gguf \
  -mtp-smoke -mtp-real-prompt \
  -prompt "Hi"

# Gemma4 QAT+MTP llama.cpp parity gate.
# Required default gate: committed MTP token/acceptance fixture + standalone
# runner + GGUF quant primitive oracles in one target.
make gemma4-mtp-parity GOTMPDIR=$PWD/.gotmp

# Expanded default commands, if the Makefile target is not available:
GOTMPDIR=$PWD/.gotmp go test ./model -run TestGemma4MTPLlamaCPPParityFixture -count=1
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/gemma4mtpparity \
  -fixture model/testdata/gemma4-mtp-llamacpp-fixture.json
GOTMPDIR=$PWD/.gotmp go test ./loader/gguf \
  -run 'TestDequantRowQ4KToZeroBlock|TestDequantRowQ4KToMatchesGGMLNibbleGroups|TestExpertMatricesQ4KGemvMatchesDequantScalar|TestDequantRowQ8_0ToMatchesScaleTimesInt8|TestQuantizeQ8_0UsesRoundAwayFromZeroWithUnroundedScale|TestDotQ4_0Q8_0MatchesAVX2Reference|TestDotQ4_0Q8_0MatchesScalarReference|TestQuantizeQ8KComputesScaleQuantsAndBlockSums|TestDequantRowQ6KToMatchesScalarReference|TestDotQ6KQ8KMatchesAVX2Reference|TestDotQ6KQ8KMatchesScalarReference' \
  -count=1 -v

# Required tagged Gemma4 GPU↔CPU compute parity gate. This uses the shared
# GPU lock, real RTX/CUDA execution, diagnostic fixtures, and bounded KV memory.
make gemma4-gpu-cpu-parity GOTMPDIR=$PWD/.gotmp

# Strict selected-logit gate. First export a llama.cpp/LiteRT reference JSON using
# the schema documented in docs/mtp-speculative.md. This target intentionally
# fails loudly if GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE is unset, and it is
# currently expected to fail until real-asset selected verifier logits match
# llama.cpp --flash-attn on 1:1:
GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE=tmp/gemma4-mtp-llamacpp-fixture.json \
GO_PHERENCE_GEMMA4_MAIN=models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf \
GO_PHERENCE_GEMMA4_MTP_DRAFTER=models/gemma4-e4b-it-google-qat-gguf/MTP/gemma-4-E4B-it-BF16-MTP.gguf \
make gemma4-mtp-strict-parity GOTMPDIR=$PWD/.gotmp

# Expanded strict commands, if the Makefile target is not available:
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/gemma4mtpparity \
  -fixture tmp/gemma4-mtp-llamacpp-fixture.json \
  -model models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf \
  -drafter models/gemma4-e4b-it-google-qat-gguf/MTP/gemma-4-E4B-it-BF16-MTP.gguf
GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE=tmp/gemma4-mtp-llamacpp-fixture.json \
GO_PHERENCE_GEMMA4_MAIN=models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf \
GO_PHERENCE_GEMMA4_MTP_DRAFTER=models/gemma4-e4b-it-google-qat-gguf/MTP/gemma-4-E4B-it-BF16-MTP.gguf \
GOTMPDIR=$PWD/.gotmp go test ./model -run TestGemma4MTPLlamaCPPParityFixture -count=1


Strict Gemma4 QAT+MTP status notes:

- `make gemma4-mtp-parity GOTMPDIR=$PWD/.gotmp` is the required green default gate.
- `make gemma4-mtp-strict-parity GOTMPDIR=$PWD/.gotmp` remains the red 1:1 selected-logit gate for the local `llama.cpp --flash-attn on` fixture.
- The strict fixture currently matches prompt/draft/verifier tokens and acceptance/bonus-token semantics, but still reports six selected verifier-logit mismatches.
- Keep `RealAssetAcceptanceParity=false`, public/default MTP generation not-ready, and full-layer verifier batch default enablement gated until the strict gate is green.

# 31B stress smoke, when VRAM headroom permits
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -gpu -gpu-layers 17 -gpu-kv-max-seq 256 \
  -model models/gemma4-31b-it-4bit \
  -mtp-drafter models/gemma4-31b-it-mtp-assistant-4bit \
  -mtp-smoke -mtp-real-prompt \
  -prompt "Hi"
```

## Benchmark snapshots

Benchmark snapshots are documentation data, not a per-change requirement. Refresh them at phase validation time with:

```bash
GOTMPDIR=$PWD/.gotmp go test ./model -run '^$' -bench 'BenchmarkCPUHot' -benchmem
```

Update `docs/performance.md` and `docs/cpu-simd-coverage.md` after capturing new snapshots.

## Hardware smoke tests

NVIDIA and Vulkan parity tests should be availability-gated. They should skip cleanly when required hardware, drivers, or opt-in environment variables are absent.

- NVIDIA smoke: validate on a CUDA-capable host with the normal backend enabled.
- Vulkan smoke: validate only after SPIR-V pipeline cache wiring lands; CPU/software Vulkan remains opt-in via `GO_PHERENCE_VULKAN_ALLOW_CPU=1`.
