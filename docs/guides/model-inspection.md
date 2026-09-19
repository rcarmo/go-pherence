# Model inspection and coverage

[Command index](commands.md) | [Runtime tuning](tuning.md)

## `minicpmvinspect` — MiniCPM-V/O metadata and prompt inspection

`cmd/minicpmvinspect` is the safe first step for OpenBMB MiniCPM-V and MiniCPM-O checkpoints. It reads `config.json`, processor/tokenizer/generation sidecars, optional safetensors headers, and optional image files, then reports prompt placeholders, image/audio special tokens, text/vision/resampler/audio plans, slice-mode planning, and readiness gates without running full tensor execution.

```bash
make minicpmv-check
make minicpmv-fixture-check
make minicpmv-inspect
make minicpmv-version
make minicpmv-support-summary
make minicpmv-capabilities
make minicpmv-pending-runtime
make minicpmv-coverage-pending
make minicpmv-assets-check
make minicpmv-fixture-path
make minicpmv-fixture-summary
make minicpmv-fixture-ready
make minicpmv-inspect-model MINICPMV_MODEL=checkpoints/minicpm-v-2.6 MINICPMV_FLAGS='-json'
make minicpmv-inspect-model MINICPMV_MODEL=model/minicpmv/testdata/minicpmo_fixture MINICPMV_AUDIO_DURATION_MS=1234

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -version

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -support-summary

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -capabilities

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -pending-runtime-steps

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -fixture-path

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -fixture-summary

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -require-fixture-ready

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -model checkpoints/minicpm-v-2.6 \
  -json

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -model checkpoints/minicpm-v-2.6 \
  -safetensors checkpoints/minicpm-v-2.6/model.safetensors \
  -require-tensors-ready \
  -require-shapes-ready

GOTMPDIR=$PWD/.gotmp go run ./cmd/minicpmvinspect \
  -model checkpoints/minicpm-o-2.6 \
  -prompt "Compare these inputs." \
  -images 2 \
  -audio-duration-ms 1234
```

Useful flags:

- `-version` — print the MiniCPM-V/O support scaffold version, runtime status, and runtime roadmap path without requiring `-model`.
- `-support-summary` — print version, runtime status, capabilities, and pending runtime steps without requiring `-model`.
- `-pending-runtime-steps` — print the remaining MiniCPM-V/O runtime implementation steps without requiring `-model`.
- `-capabilities` — print implemented-vs-pending MiniCPM-V/O capability summary without requiring `-model`; combine with `-require-capabilities-ready` for a model-free CI consistency gate.
- `-fixture-path` — print the committed MiniCPM-O metadata fixture path without requiring `-model`.
- `-fixture-summary` — print the committed MiniCPM-O expected summary without requiring `-model`.
- `-require-fixture-ready` — validate the committed MiniCPM-O fixture against its expected summary without requiring `-model`.
- `-json` — emit the full machine-readable report.
- `-safetensors PATH` — inspect one explicit safetensors file; otherwise the command tries `model.safetensors.index.json` and `model.safetensors` under `-model`.
- `-image PATH` — decode PNG/JPEG and run the configured pure-Go BCHW image preprocessing path.
- `-prompt TEXT` / `-images N` — customize image/audio placeholder preview construction.
- `-audio-duration-ms N` — estimate MiniCPM-O audio feature frames for a given duration.
- `-require-config-ready`, `-require-metadata-ready`, `-require-tensors-ready`, `-require-shapes-ready` — exit non-zero for progressively stricter scaffold readiness checks.
- `-strict` — shortcut for metadata + tensor inventory + safetensor shape readiness; does not require runtime execution.
- For tensor-only fixtures/checks without processor/tokenizer sidecars, use `-require-tensors-ready -require-shapes-ready` rather than `-strict`.
- `-require-runtime-ready` — expected to fail until full MiniCPM-V/O tensor execution lands.


## `qwen3ttsinspect` — Qwen3-TTS metadata and prompt inspection

`cmd/qwen/qwen3ttsinspect` is the safe first step for Qwen3-TTS checkpoints. It reads `config.json`, optional safetensors headers, tokenizer files, and emits shape/cache readiness without loading full inference weights into a runtime.

```bash
make qwen3tts-inspect \
  QWEN3TTS_MODEL=checkpoints/qwen3-tts-0.6b-customvoice \
  QWEN3TTS_TEXT="Hello world"
make qwen3tts-fixture-coverage \
  QWEN3TTS_MODEL=checkpoints/qwen3-tts-0.6b-customvoice

GOTMPDIR=$PWD/.gotmp go run ./cmd/qwen/qwen3ttsinspect \
  -model checkpoints/qwen3-tts-0.6b-customvoice \
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
  -model checkpoints/qwen3-tts-0.6b-customvoice \
  -fixture model/qwen3tts/testdata/customvoice_prompt_fixture.json \
  -json
```

The report includes variant/size, talker dimensions, code-predictor dimensions, tensor group readiness, runtime KV sizing, 12Hz decoder code-frame assumptions, speaker encoder presence, optional tokenized CustomVoice streams, prompt-runtime layout, optional reference coverage, runtime request planning, and combined readiness blockers. It does not synthesize audio yet; reference fixtures and CPU Talker/CodePredictor/Decoder parity are the next steps.


## `lfm2inspect` — LFM2.5 metadata and runtime-state sizing

`cmd/models/lfm2inspect` validates `lfm2_moe` config metadata, counts hybrid conv/full-attention layers, summarizes MoE settings, inspects optional safetensors headers, and reports state/cache sizing.

```bash
make lfm2-inspect LFM2_MODEL=checkpoints/lfm2.5-8b-a1b
make lfm2-fixture-coverage LFM2_MODEL=checkpoints/lfm2.5-8b-a1b

GOTMPDIR=$PWD/.gotmp go run ./cmd/models/lfm2inspect \
  -model checkpoints/lfm2.5-8b-a1b \
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
  -model checkpoints/lfm2.5-8b-a1b \
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

See [hunyuan3d-2-support.md](../models/hunyuan3d-2-support.md) for the implementation status and staged native runtime plan.


## Model coverage validation

Use this focused target for the manifest and inspector contracts. It does not prove Qwen3-TTS synthesis or LFM2 generation. Whole-tree checks are separate [host validation targets](../validation/validation-gates.md); incomplete runtime/parity gates must not be counted as implemented inference.

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
- reference/fixture gates are checked with `-fail-pending`
- parity/readiness gates are checked with `-fail-pending` and may fail while implementation is incomplete
- `cmd/models/modelcoverage`
