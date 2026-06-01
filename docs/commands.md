# Commands

## Model asset downloads

Downloaded model assets live under `models/`, which is ignored by git except for source packages. Use the helper script directly or via Make targets:

```bash
make models-list
make models-download-small
make models-download-qwen
make models-download-qwen3tts
make models-download-lfm2
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
python3 scripts/download_models.py --dry-run --group gemma4
python3 scripts/download_models.py --dry-run --group speaker
```

The downloader uses `huggingface_hub.snapshot_download`; install it with:

```bash
python3 -m pip install huggingface_hub
```

The speaker group downloads source SpeechBrain checkpoints. Convert them before use with `cmd/diarize-vtt -speaker-model`:

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
go run ./cmd/llmgen -model models/qwen3-0.6b-mlx4 -gpu -tokens 50 -prompt "The meaning of life is"
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
go run ./cmd/llmgen -model models/smollm2-135m -tokens 32 \
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

`gguf-ci-qwen36-reap` runs the focused build-only package smoke before `gguf-check-qwen36-reap`. `gguf-check-qwen36-reap` runs the validation target plus the benchmark target, including expected runtime and benchmark KV byte assertions (`245760` F32 bytes and `81920` compressed bytes for the one-token local smoke). `ggufinspect` reports REAP ratio/source, runtime readiness, KV dimensions, cache-layer counts, protected-layer counts, and full-vs-compressed KV byte estimates. `ggufsmoke -bench` runs the same generation allocator used by `GenerateWithOptions` and reports prefill/decode timing plus actual F32/compressed KV bytes for the run; `make gguf-bench-qwen36-reap` bundles the local expected token/decoded-text assertions with that benchmark. Set `GGUF_EXPECT_REAP_RATIO`, `GGUF_EXPECT_REAP_SOURCE`, `GGUF_EXPECT_ARCHITECTURE`, `GGUF_EXPECT_NAME_CONTAINS`, `GGUF_EXPECT_TENSOR_COUNT`, `GGUF_EXPECT_LAYERS`, `GGUF_EXPECT_HIDDEN_SIZE`, `GGUF_EXPECT_HEADS`, `GGUF_EXPECT_VOCAB_SIZE`, `GGUF_EXPECT_TOKENIZER_TOKENS`, `GGUF_EXPECT_BOS`, `GGUF_EXPECT_EOS`, `GGUF_EXPECT_MAX_SEQ_LEN`, `GGUF_EXPECT_FULL_ATTENTION_INTERVAL`, `GGUF_EXPECT_KV_HEADS`, `GGUF_EXPECT_HEAD_DIM`, `GGUF_EXPECT_KV_DIM`, `GGUF_EXPECT_EXPERTS`, `GGUF_EXPECT_EXPERTS_PER_TOKEN`, `GGUF_EXPECT_F32_COUNT`, `GGUF_EXPECT_Q4_K_COUNT`, `GGUF_EXPECT_Q6_K_COUNT`, `GGUF_EXPECT_CACHE_LAYERS`, `GGUF_EXPECT_PROTECTED_CACHE_LAYERS` (or the matching `ggufinspect` flags) and `GGUF_EXPECT_GENERATED`/`GGUF_EXPECT_DECODED` (or `ggufsmoke -expect-generated`/`-expect-decoded`) to make validation fail if REAP metadata/source inference, runtime shape/MoE planning, TurboQuant layer planning, synthetic compressed-KV smoke accounting, or greedy output/decoded text changes unexpectedly.

## Gemma4 MTP smoke

Standalone smoke command:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/gemma4mtpsmoke \
  -model models/gemma4-e4b-it-4bit \
  -drafter models/gemma4-e4b-mtp-drafter
```

The same experimental smoke is exposed through `llmgen`:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/llmgen \
  -gpu -gpu-layers 0 \
  -model models/gemma4-e4b-it-4bit \
  -mtp-drafter models/gemma4-e4b-mtp-drafter \
  -mtp-smoke -mtp-real-prompt \
  -prompt "Hi"
```

Use the E4B pair for local MTP development. It fits fully on the RTX 3060 and supports real prompt activation/KV capture in seconds.

31B stress path:

```bash
GOTMPDIR=$PWD/.gotmp go run ./cmd/llmgen \
  -gpu -gpu-layers 17 -gpu-kv-max-seq 256 \
  -model models/gemma4-31b-it-4bit \
  -mtp-drafter models/gemma4-31b-it-mtp-assistant-4bit \
  -mtp-smoke -mtp-real-prompt \
  -prompt "Hi"
```

`-mtp-smoke` proves the drafter/runtime seam; it is not full speculative generation yet. Full MTP generation remains pending verifier batching, adaptive draft policy, and accepted-KV commit wiring.

## Qwen3.6 native MTP triage

Recommended next Qwen checkpoint candidate:

```text
samwang0041/Qwen3.6-27B-MLX-4bit-MTP
```

It is a dense MLX affine 4-bit native-MTP checkpoint (~15.37GB) and is a better current target than the older NVFP4 public checkpoint because public NVFP4 generation remains gated.

```bash
go run ./cmd/qwenmtpmeta -model /path/to/qwen3.6-27b-mtp

go run ./cmd/qwenmtpsynth -steps 2

go run ./cmd/qwenmtpsmoke -model /path/to/qwen3.6-27b-mtp

go run ./cmd/qwen36run -model /path/to/qwen3.6-27b-mtp -prompt "Hello" -steps 1 -mtp -mtp-steps 2
```

Optional seed-variant diagnostic:

```bash
go run ./cmd/qwen36run -model /path/to/qwen3.6-27b-mtp -prompt "Hello" -steps 1 -mtp -greedy-seed
```

Sweep newline-separated prompt files:

```bash
go run ./cmd/qwen36run -model /path/to/qwen3.6-27b-mtp -sweep prompts.txt -sweep-limit 5 -steps 1 -mtp -mtp-steps 2
```

`qwenmtpmeta` inspects config/tensor metadata without entering the full model loader. `qwenmtpsynth` runs a tiny deterministic native-MTP synthetic path. `qwenmtpsmoke` loads a real native-MTP head and runs a synthetic hidden-state forward pass. `qwen36run` is the real-checkpoint CPU smoke runner.

## `qwen3ttsinspect` — Qwen3-TTS metadata and prompt inspection

`cmd/qwen3ttsinspect` is the safe first step for Qwen3-TTS checkpoints. It reads `config.json`, optional safetensors headers, tokenizer files, and emits shape/cache readiness without loading full inference weights into a runtime.

```bash
make qwen3tts-inspect \
  QWEN3TTS_MODEL=models/qwen3-tts-0.6b-customvoice \
  QWEN3TTS_TEXT="Hello world"
make qwen3tts-fixture-coverage \
  QWEN3TTS_MODEL=models/qwen3-tts-0.6b-customvoice

GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen3ttsinspect \
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
GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen3ttsinspect \
  -model models/qwen3-tts-0.6b-customvoice \
  -fixture model/qwen3tts/testdata/customvoice_prompt_fixture.json \
  -json
```

The report includes variant/size, talker dimensions, code-predictor dimensions, tensor group readiness, runtime KV sizing, 12Hz decoder code-frame assumptions, speaker encoder presence, optional tokenized CustomVoice streams, prompt-runtime layout, optional reference coverage, runtime request planning, and combined readiness blockers. It does not synthesize audio yet; reference fixtures and CPU Talker/CodePredictor/Decoder parity are the next steps.

## `lfm2inspect` — LFM2.5 metadata and runtime-state sizing

`cmd/lfm2inspect` validates `lfm2_moe` config metadata, counts hybrid conv/full-attention layers, summarizes MoE settings, inspects optional safetensors headers, and reports state/cache sizing.

```bash
make lfm2-inspect LFM2_MODEL=models/lfm2.5-8b-a1b
make lfm2-fixture-coverage LFM2_MODEL=models/lfm2.5-8b-a1b

GOTMPDIR=$PWD/.gotmp go run ./cmd/lfm2inspect \
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
GOTMPDIR=$PWD/.gotmp go run ./cmd/lfm2inspect \
  -model models/lfm2.5-8b-a1b \
  -fixture model/lfm2/testdata/lfm25_8b_a1b_metadata.json \
  -json
```

The report includes layer-pattern counts, MoE routing dimensions, conv cache settings, tensor group readiness, conv-state floats, attention KV floats/token, optional reference coverage, runtime request planning, and combined readiness blockers. It is metadata/reference scaffolding only; LFM convolution, attention, router, and expert execution remain future CPU parity work.

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
GOTMPDIR=$PWD/.gotmp go run ./cmd/modelcoverage -json
GOTMPDIR=$PWD/.gotmp go run ./cmd/modelcoverage -family qwen3_tts -pending-only
GOTMPDIR=$PWD/.gotmp go run ./cmd/modelcoverage -family qwen3_tts -references-only -pending-only
GOTMPDIR=$PWD/.gotmp go run ./cmd/modelcoverage -runtime-only -pending-only
GOTMPDIR=$PWD/.gotmp go run ./cmd/modelcoverage -execution-only -pending-only
GOTMPDIR=$PWD/.gotmp go run ./cmd/modelcoverage -parity-only -pending-only
GOTMPDIR=$PWD/.gotmp go run ./cmd/modelcoverage -readiness-only -pending-only
```

This runs tests and vet for:

- `loader/safetensors`
- `model/qwen3tts`
- `model/lfm2`
- `cmd/qwen3ttsinspect`
- `cmd/lfm2inspect`
- reference/fixture coverage has no pending manifest gates
- parity/readiness coverage has no pending manifest gates
- `cmd/modelcoverage`

## `specbench` / `speccheck`

```bash
go run ./cmd/specbench -model models/smollm2-135m \
  -prompt-file prompts.txt -tokens 16 -repeat 3 \
  -speculative-proposer prompt -csv specbench.csv

go run ./cmd/speccheck -model models/smollm2-135m \
  -prompt-file prompts.txt -tokens 16 \
  -proposers prompt,repeat-last,none
```

`specbench` emits normal/speculative rows with output parity, speedup, verifier backend, proposer, acceptance/fallback counters, emitted tokens, tokens/step, average proposal length, and aggregate total rows. `speccheck` emits JSON and exits non-zero on mismatch; use `-write-golden` / `-golden` to save and compare baselines.

## `diarize-vtt` — large-v3 translated WebVTT

`cmd/diarize-vtt` is the current long-form audio command. It defaults to Whisper large-v3 translation, VAD-packed chunks, progressive writes, and resume support:

```bash
go run ./cmd/diarize-vtt \
  -input meeting.m4a \
  -output meeting.vtt \
  -language es
```

Useful flags:

- `-task translate|transcribe` — default `translate`.
- `-language CODE` — source language prompt (`pt` default; use `es`, `en`, etc. as needed).
- `-workers N` — default `min(16, runtime.NumCPU())`; local stress testing found 16 best and 20 regressed.
- `-chunk 10 -overlap 1 -vad-pack=true` — default VAD-packed chunk profile.
- `-max-tokens 40 -tokens-per-sec 4` — tuned decoder token budget.
- `-progressive=true -resume=true` — preserve and resume partial VTTs.
- `-gpu=true` — GPU-assisted encoder, cross-KV precompute, and LM head.
- `-speaker-model PATH` — optional converted ECAPA safetensors speaker embedding model.
- `-speaker-threshold 0.3` — cosine similarity threshold for speaker clustering.

Current limitations: speaker labels remain a single-speaker fallback unless `-speaker-model` points to converted ECAPA weights; `GO_PHERENCE_WHISPER_GPU_DECODER_MLP=1` and `GO_PHERENCE_WHISPER_GPU_CROSS_ATTN=1` are experimental and slower on the current stress sample. See [whisper-diarize-vtt.md](whisper-diarize-vtt.md).

## `speakercheck` — speaker-only ECAPA validation

Use this to validate VAD → ECAPA embeddings → clustering without loading Whisper. WAV files are read directly; other audio formats such as M4A are decoded through `ffmpeg` when available.

```bash
go run ./cmd/speakercheck \
  -input testdata/jfk.wav \
  -speaker-model models/speaker-ecapa-voxceleb.safetensors \
  -threshold 0.3 \
  -context 0.5
```

For long recordings, spot-check a short window:

```bash
go run ./cmd/speakercheck \
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
go run ./cmd/llmchat -model models/gemma4-e2b-mlx4 -gpu -n 256
```

## `llmserver`

```bash
go run ./cmd/llmserver -model models/gemma4-e2b-mlx4 -gpu -listen :8080
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma4-e2b-mlx4","messages":[{"role":"user","content":"Hello"}]}'
```
