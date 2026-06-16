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
# The parity target fails loudly if local turbo weights/tokenizer/JFK audio are missing.
make whisper-turbo-parity
make whisper-turbo-check

# Equivalent explicit form:
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

# DiffusionGemma runnable llama.cpp GGUF golden/probe gate. This is fixture-only
# and does not require local 48 GiB shards or CUDA; it locks prompt IDs,
# reference response IDs, current Go-vs-llama mismatch metadata, first-step
# top-logit probe values, phase-aligned row28 structural trace parity
# (ffn_post_norm/l_out across all text layers), plus Q4_K/Q8_0/Q5_0 direct
# expert row-dot parity against a scalar dequantized CPU oracle.
make diffusiongemma-golden-gate

# Diagnostic package import/build boundary, when local diagnostic assets permit it
GOTMPDIR=$PWD/.gotmp go test -tags diagnostic ./model/gemma4

# Gemma4 31B packed MTP smoke, when local ignored model assets are present
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -model models/gemma4-31b-it-4bit \
  -mtp-drafter models/gemma4-31b-it-mtp-assistant-4bit \
  -mtp-smoke \
  -prompt "Hello"

# Recommended real-prompt Gemma4 E4B smoke on the local RTX 3060 profile
# If the main verifier snapshot is missing, fetch it with:
#   python3 scripts/download_models.py --only gemma4-e4b-it-4bit
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -gpu -gpu-layers 0 \
  -model models/gemma4-e4b-it-4bit \
  -mtp-drafter models/gemma4-e4b-mtp-drafter \
  -mtp-smoke -mtp-real-prompt \
  -prompt "Hi"

# Gemma4 QAT+MTP llama.cpp parity gate.
# Default committed fixture: always runs, validating token/acceptance parity when
# local GGUF assets are present and a trimmed token/acceptance contract otherwise.
GOTMPDIR=$PWD/.gotmp go test ./model -run TestGemma4MTPLlamaCPPParityFixture -count=1
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/gemma4mtpparity \
  -fixture model/testdata/gemma4-mtp-llamacpp-fixture.json

# GGUF quant primitive oracles used by the Gemma4 QAT+MTP path.
GOTMPDIR=$PWD/.gotmp go test ./loader/gguf \
  -run 'TestDequantRowQ4KToZeroBlock|TestDequantRowQ4KToMatchesGGMLNibbleGroups|TestQuantizeQ8_0UsesRoundAwayFromZeroWithUnroundedScale|TestDotQ4_0Q8_0MatchesScalarReference|TestDotQ6KQ8KMatchesScalarReference' \
  -count=1 -v

# Strict selected-logit gate. First export a llama.cpp/LiteRT reference JSON using
# the schema documented in docs/mtp-speculative.md, then run both the standalone
# token/logit report and the Go test selected-logit gate:
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/gemma4mtpparity \
  -fixture tmp/gemma4-mtp-llamacpp-fixture.json \
  -model models/gemma4-e4b-it-4bit \
  -drafter models/gemma4-e4b-mtp-drafter
GO_PHERENCE_GEMMA4_MTP_LLAMA_CPP_FIXTURE=tmp/gemma4-mtp-llamacpp-fixture.json \
GO_PHERENCE_GEMMA4_MAIN=models/gemma4-e4b-it-4bit \
GO_PHERENCE_GEMMA4_MTP_DRAFTER=models/gemma4-e4b-mtp-drafter \
GOTMPDIR=$PWD/.gotmp go test ./model -run TestGemma4MTPLlamaCPPParityFixture -count=1

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
