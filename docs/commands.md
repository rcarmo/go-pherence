# Commands

## Model asset downloads

Downloaded model assets live under `models/`, which is ignored by git except for source packages. Use the helper script directly or via Make targets:

```bash
make models-list
make models-download-small
make models-download-qwen
make models-download-qwen3tts
make models-download-lfm2
make models-download-minicpmv
make models-download-minicpmo
make models-download-gemma4
make models-download-speaker
make models-download-one MODEL=qwen3.6-27b-mlx4-mtp
```

Forward extra options through `MODEL_DOWNLOAD_FLAGS`:

```bash
make models-download-one MODEL=qwen3.6-27b-mlx4-mtp MODEL_DOWNLOAD_FLAGS='--force'
make models-list MODEL_DOWNLOAD_FLAGS='--group qwen'
make models-list MODEL_DOWNLOAD_FLAGS='--group qwen3tts'
make models-list MODEL_DOWNLOAD_FLAGS='--group lfm2'
make models-list MODEL_DOWNLOAD_FLAGS='--group minicpmv'
make models-list MODEL_DOWNLOAD_FLAGS='--group minicpmo'
python3 scripts/download_models.py --dry-run --group gemma4
python3 scripts/download_models.py --dry-run --group speaker
```

MiniCPM-V/O checkpoints are gated on some Hugging Face mirrors. `make models-download-minicpmv` fetches the combined MiniCPM-V/O group, while `make models-download-minicpmo` fetches only MiniCPM-O.

The downloader uses `huggingface_hub.snapshot_download`; install it with:

```bash
python3 -m pip install huggingface_hub
```

The speaker group downloads source SpeechBrain checkpoints. Convert them before use with `cmd/audio/diarize-vtt -speaker-model`:

```bash
python3 -m pip install torch safetensors
python3 scripts/convert_speechbrain_ecapa.py \
  --checkpoint models/speechbrain-ecapa-voxceleb/embedding_model.ckpt \
  --output models/speaker-ecapa-voxceleb.safetensors \
  --dump-keys
```

For gated repositories, set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`. If an upstream repo is renamed, override it without editing the script:

```bash
python3 scripts/download_models.py --only gemma4-e4b-it-4bit --repo gemma4-e4b-it-4bit=org/repo
```

## `llmgen` — one-shot generation

```bash
go run ./cmd/llm/llmgen -model models/qwen3-0.6b-mlx4 -gpu -tokens 50 -prompt "The meaning of life is"
```

Useful flags:

- `-gpu` — enable NVIDIA GPU backend when available.
- `-gpu-layers N` — hybrid CPU/GPU inference (`0` means all possible layers on GPU).
- `-gpu-kv-max-seq N` — GPU KV horizon; lower values fit more layers for prompt/MTP smokes.
- `-eager-load` — pre-fault mmap'd safetensors weights at startup.
- `-turbo-quant` — CPU-only TurboQuant KV compression.
- `-speculative` — stock-weight speculative scaffold on CPU backend.

CPU speculative scaffold example:

```bash
go run ./cmd/llm/llmgen -model models/smollm2-135m -tokens 32 \
  -prompt "abc abc abc abc" \
  -speculative -speculative-proposer prompt -speculative-debug
```

The current stock speculative backend is `replay`, a correctness scaffold that reuses the CPU generator and can be slower. Available proposer choices are `prompt`, `repeat-last`, and `none`; `-speculative-min-proposal` gates tiny proposals.

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

## Gemma4 MTP smoke

Standalone smoke command:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/gemma4mtpsmoke \
  -model models/gemma4-e4b-it-4bit \
  -drafter models/gemma4-e4b-mtp-drafter
```

The same experimental smoke is exposed through `llmgen`:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -gpu -gpu-layers 0 \
  -model models/gemma4-e4b-it-4bit \
  -mtp-drafter models/gemma4-e4b-mtp-drafter \
  -mtp-smoke -mtp-real-prompt \
  -prompt "Hi"
```

Use the E4B pair for local MTP development. It fits fully on the RTX 3060 and supports real prompt activation/KV capture in seconds.

31B stress path:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -gpu -gpu-layers 17 -gpu-kv-max-seq 256 \
  -model models/gemma4-31b-it-4bit \
  -mtp-drafter models/gemma4-31b-it-mtp-assistant-4bit \
  -mtp-smoke -mtp-real-prompt \
  -prompt "Hi"
```

`-mtp-smoke` proves the drafter/runtime seam and prints `mtp_graph_capabilities`/public-generation blockers. Experimental graph-backed MTP generation is available on the CPU verifier path:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/llm/llmgen \
  -model models/gemma4-e4b-it-4bit \
  -mtp-drafter models/gemma4-e4b-mtp-drafter \
  -mtp-generate \
  -tokens 16 \
  -prompt "Hi"
```

`-mtp-generate` builds a prompt context, maps external KV into the q-only drafter, and invokes `GenerateMTPGraphFromPromptContext` with adaptive draft policy. It remains experimental until real-asset verifier-batch parity/default enablement is complete; regular generation is unchanged unless this flag is set.

## Qwen3.6 native MTP triage

Recommended next Qwen checkpoint candidate:

```text
samwang0041/Qwen3.6-27B-MLX-4bit-MTP
```

It is a dense MLX affine 4-bit native-MTP checkpoint (~15.37GB) and is a better current target than the older NVFP4 public checkpoint because public NVFP4 generation remains gated.

```bash
go run ./cmd/qwen/qwenmtpmeta -model /path/to/qwen3.6-27b-mtp

go run ./cmd/qwen/qwenmtpsynth -steps 2

go run ./cmd/qwen/qwenmtpsmoke -model /path/to/qwen3.6-27b-mtp

go run ./cmd/qwen/qwen36run -model /path/to/qwen3.6-27b-mtp -prompt "Hello" -steps 1 -mtp -mtp-steps 2
```

Optional seed-variant diagnostic:

```bash
go run ./cmd/qwen/qwen36run -model /path/to/qwen3.6-27b-mtp -prompt "Hello" -steps 1 -mtp -greedy-seed
```

Sweep newline-separated prompt files:

```bash
go run ./cmd/qwen/qwen36run -model /path/to/qwen3.6-27b-mtp -sweep prompts.txt -sweep-limit 5 -steps 1 -mtp -mtp-steps 2
```

`qwenmtpmeta` inspects config/tensor metadata without entering the full model loader. `qwenmtpsynth` runs a tiny deterministic native-MTP synthetic path. `qwenmtpsmoke` loads a real native-MTP head and runs a synthetic hidden-state forward pass. `qwen36run` is the real-checkpoint CPU smoke runner.

## `minicpmvinspect` — MiniCPM-V/O metadata and prompt inspection

`cmd/minicpmvinspect` is the safe first step for OpenBMB MiniCPM-V and MiniCPM-O checkpoints. It reads `config.json`, processor/tokenizer/generation sidecars, optional safetensors headers, and optional image files, then reports prompt placeholders, image/audio special tokens, text/vision/resampler/audio plans, slice-mode planning, and readiness gates without running full tensor execution.

```bash
make minicpmv-check
make minicpmv-fixture-check
make minicpmv-inspect
make minicpmv-version
make minicpmv-capabilities
make minicpmv-inspect-model MINICPMV_MODEL=models/minicpm-v-2.6 MINICPMV_FLAGS='-json'

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -version

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -capabilities

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -fixture-path

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -model models/minicpm-v-2.6 \
  -json

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -model models/minicpm-v-2.6 \
  -safetensors models/minicpm-v-2.6/model.safetensors \
  -require-tensors-ready \
  -require-shapes-ready

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -model models/minicpm-o-2.6 \
  -prompt "Compare these inputs." \
  -images 2
```

Useful flags:

- `-version` — print the MiniCPM-V/O support scaffold version and runtime status without requiring `-model`.
- `-capabilities` — print implemented-vs-pending MiniCPM-V/O capability summary without requiring `-model`; combine with `-require-capabilities-ready` for a model-free CI consistency gate.
- `-fixture-path` — print the committed MiniCPM-O metadata fixture path without requiring `-model`.
- `-json` — emit the full machine-readable report.
- `-safetensors PATH` — inspect one explicit safetensors file; otherwise the command tries `model.safetensors.index.json` and `model.safetensors` under `-model`.
- `-image PATH` — decode PNG/JPEG and run the configured pure-Go BCHW image preprocessing path.
- `-prompt TEXT` / `-images N` — customize image/audio placeholder preview construction.
- `-require-config-ready`, `-require-metadata-ready`, `-require-tensors-ready`, `-require-shapes-ready` — exit non-zero for progressively stricter scaffold readiness checks.
- `-strict` — shortcut for metadata + tensor inventory + safetensor shape readiness; does not require runtime execution.
- `-require-runtime-ready` — expected to fail until full MiniCPM-V/O tensor execution lands.

## `qwen3ttsinspect` — Qwen3-TTS metadata and prompt inspection

`cmd/qwen/qwen3ttsinspect` is the safe first step for Qwen3-TTS checkpoints. It reads `config.json`, optional safetensors headers, tokenizer files, and emits shape/cache readiness without loading full inference weights into a runtime.

```bash
make qwen3tts-inspect \
  QWEN3TTS_MODEL=models/qwen3-tts-0.6b-customvoice \
  QWEN3TTS_TEXT="Hello world"
make qwen3tts-fixture-coverage \
  QWEN3TTS_MODEL=models/qwen3-tts-0.6b-customvoice

GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen/qwen3ttsinspect \
  -model models/qwen3-tts-0.6b-customvoice \
  -text "Hello world" \
  -speaker ryan \
  -language en \
  -json
```

Useful flags:

- `-json` — emit the full machine-readable report.
- `-strict` — exit non-zero when safetensors are present and tensor readiness or shape validation fails, or when requested conditioning validation fails.
- `-safetensors PATH` — inspect one explicit safetensors file; otherwise the command tries `model.safetensors.index.json` and `model.safetensors` under `-model`.
- `-text TEXT` — load tokenizer files from `-model`, tokenize `TEXT`, and build the deterministic CustomVoice text/codec control streams.
- `-first-text-id ID` — build only the fixed CustomVoice prefix around a known first tokenizer ID.
- `-speaker NAME` / `-language CODE` — select CustomVoice control tokens for the prompt probe.
- `-reference-audio PATH` — mark Base/reference-audio conditioning as present for capability validation.
- `-voice-prompt TEXT` — provide VoiceDesign conditioning text for capability validation.
- `-fixture PATH` — load a compact reference fixture and report which prompt/semantic/acoustic/WAV parity anchors are present or still missing.
- `-require-complete-fixture` — with `-fixture`, exit non-zero unless all reference anchors are present.
- `-require-numeric-parity` — with `-fixture`, exit non-zero while reference checksums are still placeholder values.
- `-require-runtime` — exit non-zero until Talker, CodePredictor, and Decoder12Hz runtime execution are implemented.
- `-require-ready` — exit non-zero until runtime execution and numeric parity are both ready.

Fixture coverage example:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen/qwen3ttsinspect \
  -model models/qwen3-tts-0.6b-customvoice \
  -fixture model/qwen3tts/testdata/customvoice_prompt_fixture.json \
  -json
```

The report includes variant/size, talker dimensions, code-predictor dimensions, tensor group readiness, runtime KV sizing, 12Hz decoder code-frame assumptions, speaker encoder presence, optional tokenized CustomVoice streams, prompt-runtime layout, optional reference coverage, runtime request planning, and combined readiness blockers. It does not synthesize audio yet; reference fixtures and CPU Talker/CodePredictor/Decoder parity are the next steps.

## `lfm2inspect` — LFM2.5 metadata and runtime-state sizing

`cmd/models/lfm2inspect` validates `lfm2_moe` config metadata, counts hybrid conv/full-attention layers, summarizes MoE settings, inspects optional safetensors headers, and reports state/cache sizing.

```bash
make lfm2-inspect LFM2_MODEL=models/lfm2.5-8b-a1b
make lfm2-fixture-coverage LFM2_MODEL=models/lfm2.5-8b-a1b

GOTMPDIR=$PWD/.gotmp go run ./cmd/models/lfm2inspect \
  -model models/lfm2.5-8b-a1b \
  -json
```

Useful flags:

- `-json` — emit the full machine-readable report.
- `-strict` — exit non-zero when safetensors are present and tensor readiness or shape validation fails.
- `-safetensors PATH` — inspect one explicit safetensors file; otherwise the command tries `model.safetensors.index.json` and `model.safetensors` under `-model`.
- `-fixture PATH` — load compact reference metadata and report which config/tensor/reference parity anchors are present or still missing.
- `-require-complete-fixture` — with `-fixture`, exit non-zero unless all reference anchors are present.
- `-require-numeric-parity` — with `-fixture`, exit non-zero while reference checksums are still placeholder values.
- `-require-runtime` — exit non-zero until LFM2 generation runtime execution is implemented.
- `-require-ready` — exit non-zero until runtime execution and numeric parity are both ready.

Fixture coverage example:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/lfm2inspect \
  -model models/lfm2.5-8b-a1b \
  -fixture model/lfm2/testdata/lfm25_8b_a1b_metadata.json \
  -json
```

The report includes layer-pattern counts, MoE routing dimensions, conv cache settings, tensor group readiness, conv-state floats, attention KV floats/token, optional reference coverage, runtime request planning, and combined readiness blockers. It is metadata/reference scaffolding only; LFM convolution, attention, router, and expert execution remain future CPU parity work.

## Hunyuan3D fixture and runtime-scaffold commands

Hunyuan3D is a diffusion/vision/3D pipeline, not an LLM decode path. The current Go implementation can parse configs, inspect/check safetensors groups, run native image preprocessing and FlowMatch scheduling scaffolds, compare fixture JSON, and exercise shared CPU/SIMD ViT primitives. End-to-end native GLB generation is still blocked on conditioner tensor binding, Hunyuan3DDiT, ShapeVAE, and mesh kernels.

```bash
# Environment/readiness report for optional Python fixture generation.
make hunyuan3d-fixture-env \
  HUNYUAN3D_IMAGE=testdata/hunyuan3d/seahorse_rgba.png

# Hugging Face metadata/header inventory without downloading full tensor payloads.
make hunyuan3d-inventory

# Local YAML + safetensors group inspection once checkpoint payloads are present.
make hunyuan3d-inspect \
  HUNYUAN3D_CONFIG=/path/to/hunyuan3d-dit-v2-mini/config.yaml \
  HUNYUAN3D_CHECKPOINT=/path/to/model.fp16.safetensors

# Optional Python upstream seahorse GLB generation helper, dependency/VRAM gated.
make hunyuan3d-seahorse
```

Fixture targets for parity work:

```bash
make hunyuan3d-image-fixture
make hunyuan3d-conditioner-fixture \
  HUNYUAN3D_CONFIG=/path/to/config.yaml \
  HUNYUAN3D_CHECKPOINT=/path/to/model.fp16.safetensors \
  HUNYUAN3D_IMAGE=/path/to/image.png
make hunyuan3d-denoiser-fixture ...
make hunyuan3d-lowstep-fixture ...
make hunyuan3d-mesh-fixture ...
```

See [hunyuan3d-2-support.md](hunyuan3d-2-support.md) for the implementation status and staged native runtime plan.

## Model coverage validation

Use the focused target below while Qwen3-TTS and LFM2 are still metadata/fixture/inspector work. It avoids unrelated experimental packages that currently fail whole-tree dry compiles.

```bash
make test-model-coverage
```

Show the current manifest summary with; text, JSON, Markdown, and CSV output include per-family category counts and completion percentages for reference, runtime, execution, parity, and readiness gates. The Make targets bootstrap `$(GOTMPDIR)` before invoking Go. `make test-model-coverage` also enforces `MODEL_COVERAGE_MIN_PERCENT` (default `90`) through `modelcoverage -min-percent` and compares the generated snapshot against `docs/model-coverage-snapshot.md`.

```bash
make model-coverage
make model-coverage-json
make model-coverage-markdown
make model-coverage-csv
make model-coverage-snapshot
make model-coverage-snapshot-file
make model-coverage-snapshot-check
make model-coverage-runtime-roadmap
make model-coverage-runtime-roadmap-json
make model-coverage-next-runtime
make model-coverage-next-runtime-json
# add -blocker-package model/qwen3tts, model/lfm2, or backends/nvidia, or -blocker-kind cpu/nvidia/streaming, to scope roadmap/next-runtime output
# emits phase/kind-numbered, dependency-ordered runtime blocker checklists with package/fixture hints, short descriptions, prerequisites, and validation hints
make model-coverage-pending MODEL_COVERAGE_FAMILY=qwen3_tts
make model-coverage-references-pending
make model-coverage-runtime-pending
make model-coverage-execution-pending
make model-coverage-parity-pending
make model-coverage-readiness-pending
make model-coverage-references-gate
make model-coverage-runtime-gate
make model-coverage-execution-gate
make model-coverage-parity-gate
make model-coverage-readiness-gate
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/modelcoverage -json
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/modelcoverage -family qwen3_tts -pending-only
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/modelcoverage -family qwen3_tts -references-only -pending-only
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/modelcoverage -runtime-only -pending-only
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/modelcoverage -execution-only -pending-only
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/modelcoverage -parity-only -pending-only
GOTMPDIR=$PWD/.gotmp go run ./cmd/models/modelcoverage -readiness-only -pending-only
```

This runs tests and vet for:

- `loader/safetensors`
- `model/qwen3tts`
- `model/lfm2`
- `cmd/qwen/qwen3ttsinspect`
- `cmd/models/lfm2inspect`
- reference/fixture coverage has no pending manifest gates
- parity/readiness coverage has no pending manifest gates
- `cmd/models/modelcoverage`

## `specbench` / `speccheck`

```bash
go run ./cmd/llm/specbench -model models/smollm2-135m \
  -prompt-file prompts.txt -tokens 16 -repeat 3 \
  -speculative-proposer prompt -csv specbench.csv

go run ./cmd/llm/speccheck -model models/smollm2-135m \
  -prompt-file prompts.txt -tokens 16 \
  -proposers prompt,repeat-last,none
```

`specbench` emits normal/speculative rows with output parity, speedup, verifier backend, proposer, acceptance/fallback counters, emitted tokens, tokens/step, average proposal length, and aggregate total rows. `speccheck` emits JSON and exits non-zero on mismatch; use `-write-golden` / `-golden` to save and compare baselines.

## `whisper` — standalone turbo STT/translation

`cmd/audio/whisper` is the simpler single-command Whisper entry point. It defaults to local `openai/whisper-large-v3-turbo` weights and accepts WAV directly or other audio formats through `ffmpeg`.

```bash
go run ./cmd/audio/whisper \
  -audio meeting.m4a \
  -task translate \
  -language en
```

Useful flags:

- `-model PATH` — default `models/whisper-large-v3-turbo-hf/model.safetensors`.
- `-size turbo|large-v3|...` — default `turbo`.
- `-task transcribe|translate` — default `transcribe`; use `translate` for English translation.
- `-language CODE` — default `en`; for turbo translated English, keep `en`.
- `-timestamps` — emit timestamped segments; with `-output something.vtt`, writes WebVTT.
- `-diarize` — add speaker labels when `-timestamps` is enabled and the speaker model is available.
- `-chunk N -chunk-workers N` — long-form windowing controls; simple no-timestamp mode also chunks long inputs instead of sending over-length audio to the encoder.
- `-max-tokens N` — cap decoder output for smokes/benchmarks.
- `-gpu` — opt into the GPU-assisted encoder/LM-head path when CUDA SGEMM is available; falls back to CPU/SIMD otherwise.
- `-gpu-graph` — enable `GO_PHERENCE_WHISPER_GPU_GRAPH=1` for the full opt-in Whisper GPU graph surface; implies `-gpu` and remains parity/fallback guarded.

Quick turbo smoke:

```bash
go run ./cmd/audio/whisper -audio testdata/jfk.wav -task translate -language en -max-tokens 4
```

## `diarize-vtt` — turbo translated WebVTT

`cmd/audio/diarize-vtt` is the current long-form audio command. It now defaults to Whisper large-v3-turbo translation, VAD-packed chunks, progressive writes, and resume support:

```bash
go run ./cmd/audio/diarize-vtt \
  -input meeting.m4a \
  -output meeting.vtt
```

Useful flags:

- `-task translate|transcribe` — default `translate`.
- `-language CODE` — Whisper language prompt (`en` default for turbo English translation; use `pt`/`es` for source-language prompts or full large-v3 behavior).
- `-workers N` — default `min(16, runtime.NumCPU())`; local stress testing found 16 best and 20 regressed.
- `-chunk 10 -overlap 1 -vad-pack=true` — default VAD-packed chunk profile.
- `-max-tokens 40 -tokens-per-sec 4` — tuned decoder token budget.
- `-progressive=true -resume=true` — preserve and resume partial VTTs.
- `-gpu=true` — GPU-assisted encoder and LM head when CUDA SGEMM is available; cross-K/V precompute remains separately gated by `-gpu-graph` or `GO_PHERENCE_WHISPER_GPU_CROSS_KV=1`.
- `-gpu-graph=true` — enable `GO_PHERENCE_WHISPER_GPU_GRAPH=1` for all currently wired opt-in Whisper GPU graph surfaces; implies `-gpu=true`.
- `-speaker-model PATH` — optional converted ECAPA safetensors speaker embedding model.
- `-speaker-threshold 0.3` — cosine similarity threshold for speaker clustering.

Current limitations: speaker labels remain a single-speaker fallback unless `-speaker-model` points to converted ECAPA weights; `GO_PHERENCE_WHISPER_GPU_GRAPH=1`, `GO_PHERENCE_WHISPER_GPU_DECODER_MLP=1`, and `GO_PHERENCE_WHISPER_GPU_CROSS_ATTN=1` are experimental and slower on the current stress sample. See [whisper-diarize-vtt.md](whisper-diarize-vtt.md).

## `speakercheck` — speaker-only ECAPA validation

Use this to validate VAD → ECAPA embeddings → clustering without loading Whisper. WAV files are read directly; other audio formats such as M4A are decoded through `ffmpeg` when available.

```bash
go run ./cmd/audio/speakercheck \
  -input testdata/jfk.wav \
  -speaker-model models/speaker-ecapa-voxceleb.safetensors \
  -threshold 0.3 \
  -context 0.5
```

For long recordings, spot-check a short window:

```bash
go run ./cmd/audio/speakercheck \
  -input testdata/podcast.wav \
  -speaker-model models/speaker-ecapa-voxceleb.safetensors \
  -start 300 \
  -duration 30 \
  -sims=false
```

It prints VAD segment timings, assigned speaker labels, speaker counts, and optional pairwise cosine similarities. Add `-json` to emit a machine-readable report. Add `-expect 1,1,2,2` to score a labeled fixture by exact label accuracy and pairwise same/different agreement; text mode exits non-zero when pairwise score is below 1.

Run a repeatable labeled suite with:

```bash
python3 scripts/speakercheck_suite.py testdata/speakercheck_suite.json
```

## `llmchat`

```bash
go run ./cmd/llm/llmchat -model models/gemma4-e2b-mlx4 -gpu -n 256
```

## `llmserver`

```bash
go run ./cmd/llm/llmserver -model models/gemma4-e2b-mlx4 -gpu -listen :8080
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-e2b-mlx4","messages":[{"role":"user","content":"Hello"}]}'
```

For llama.cpp-compatible TurboQuant policy diagnostics on CPU/server paths, pass the cache flags and inspect `/health`:

```bash
go run ./cmd/llm/llmserver \
  -model models/qwen3-moe \
  -listen :8080 \
  -cache-type-k turbo4 \
  -cache-type-v turbo2 \
  -kv-residual-window 128

curl -s http://localhost:8080/health | jq '.turboquant, .reap'
```

The health payload reports native go-pherence interpretation of those policy names: key/value bits, KV shape, full/estimated/saved bytes, ratio, KV layer count, protected-layer count, and REAP summary/source when the loaded model has REAP enabled.


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
