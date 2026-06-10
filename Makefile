TMPDIR ?= /workspace/tmp
GOTMPDIR ?= /workspace/tmp
PYTHON ?= python3
MODELS_DIR ?= models
MODEL ?=
MODEL_DOWNLOAD_FLAGS ?=
export TMPDIR GOTMPDIR

.PHONY: all build test test-cpu test-model-coverage model-coverage-tmpdir model-coverage model-coverage-json model-coverage-markdown model-coverage-csv model-coverage-snapshot model-coverage-snapshot-file model-coverage-snapshot-check model-coverage-runtime-roadmap model-coverage-runtime-roadmap-json model-coverage-next-runtime model-coverage-next-runtime-json model-coverage-pending model-coverage-references-pending model-coverage-runtime-pending model-coverage-execution-pending model-coverage-parity-pending model-coverage-readiness-pending model-coverage-references-gate model-coverage-runtime-gate model-coverage-execution-gate model-coverage-parity-gate model-coverage-readiness-gate clean server chat gen vet models-list models-download models-download-small models-download-qwen models-download-qwen3tts models-download-lfm2 models-download-gemma4 models-download-speaker models-download-one gguf-inspect gguf-smoke gguf-bench gguf-turboquant-smoke gguf-validate gguf-check gguf-ci gguf-inspect-qwen36-reap gguf-smoke-qwen36-reap gguf-validate-qwen36-reap gguf-bench-qwen36-reap gguf-check-qwen36-reap gguf-ci-qwen36-reap qwen3tts-inspect qwen3tts-fixture-coverage lfm2-inspect lfm2-fixture-coverage hunyuan3d-fixture-env hunyuan3d-inventory hunyuan3d-inspect hunyuan3d-image-fixture hunyuan3d-conditioner-fixture hunyuan3d-denoiser-fixture hunyuan3d-lowstep-fixture hunyuan3d-mesh-fixture trellis2-fixture-env trellis2-inventory trellis2-lowstep-fixture trellis2-ovoxel-inspect whisper whisper-k3 speaker-weights

all: build

build: gen server chat

gen:
	go build -o bin/llmgen ./cmd/llm/llmgen

server:
	go build -o bin/llmserver ./cmd/llm/llmserver

chat:
	go build -o bin/llmchat ./cmd/llm/llmchat

# Whisper speech-to-text. The optimized RVV + SpaceMIT IME (int8) kernels are
# gated by //go:build riscv64 and selected at runtime via CPU feature detection,
# so a native riscv64 build picks them up automatically. See
# docs/whisper-riscv-optimization.md for the optimization details and the
# WHISPER_* runtime tunables.
whisper:
	go build -o bin/whisper ./cmd/audio/whisper

# Build for the SpaceMIT K1/K3 (MilkV Jupiter 2: 8x X60 RISC-V, RVV 1.0 + IME
# integer matrix engine). Forces GOARCH=riscv64 so it can be cross-compiled from
# an x86 host as well as built natively on the board; CGO disabled for a static
# binary. Run with WHISPER_INT8=1 for the full int8 IME pipeline.
whisper-k3:
	mkdir -p $(GOTMPDIR)
	CGO_ENABLED=0 GOOS=linux GOARCH=riscv64 go build -o bin/whisper-k3 ./cmd/audio/whisper
	@echo "Built bin/whisper-k3 (riscv64, RVV + IME int8)."
	@echo "Resident run:  WHISPER_INT8=1 WHISPER_THREADS=4 bin/whisper-k3 -model <model.safetensors> -size large-v3 -audio <file.wav>"

SPEAKER_CKPT ?= $(MODELS_DIR)/speechbrain-ecapa-voxceleb/embedding_model.ckpt
SPEAKER_SAFETENSORS ?= $(MODELS_DIR)/speaker-ecapa-voxceleb.safetensors
SPEAKER_CKPT_URL ?= https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt

# Download + convert the SpeechBrain ECAPA-TDNN speaker-embedding weights WITHOUT
# torch (works on the RISC-V board; needs only python3 + numpy). Produces the
# safetensors that `whisper -diarize` and cmd/audio/speakercheck consume. The full
# torch-based converter (scripts/convert_speechbrain_ecapa.py) remains for hosts
# that have torch installed.
speaker-weights:
	mkdir -p $(dir $(SPEAKER_CKPT))
	[ -f $(SPEAKER_CKPT) ] || curl -fsSL -o $(SPEAKER_CKPT) "$(SPEAKER_CKPT_URL)"
	$(PYTHON) scripts/ckpt_to_safetensors_numpy.py --checkpoint $(SPEAKER_CKPT) --output $(SPEAKER_SAFETENSORS)
	@echo "Wrote $(SPEAKER_SAFETENSORS). Diarize with: whisper -timestamps -diarize -audio <file.wav> ..."

test:
	go test -count=1 -timeout=120s ./loader/... ./model/... ./models/bert/... ./backends/nvidia/... ./backends/placement/... ./backends/simd/... ./backends/vulkan/... ./runtime/... ./tensor/...

test-cpu:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_VULKAN_ALLOW_CPU=0 go test -count=1 -timeout=120s ./loader/... ./model/... ./models/bert/... ./backends/nvidia/... ./backends/placement/... ./backends/simd/... ./backends/vulkan/... ./runtime/... ./tensor/...

test-model-coverage: model-coverage-tmpdir
	go test -count=1 -timeout=120s ./docs ./loader/safetensors ./model/qwen3tts ./model/lfm2 ./cmd/qwen/qwen3ttsinspect ./cmd/models/lfm2inspect ./cmd/models/modelcoverage
	go vet ./docs ./loader/safetensors ./model/qwen3tts ./model/lfm2 ./cmd/qwen/qwen3ttsinspect ./cmd/models/lfm2inspect ./cmd/models/modelcoverage
	go run ./cmd/models/modelcoverage -references-only -fail-pending
	go run ./cmd/models/modelcoverage -parity-only -fail-pending
	go run ./cmd/models/modelcoverage -readiness-only -fail-pending
	go run ./cmd/models/modelcoverage -min-percent $(MODEL_COVERAGE_MIN_PERCENT)
	$(MAKE) model-coverage-snapshot-check

MODEL_COVERAGE_FAMILY ?=
MODEL_COVERAGE_MIN_PERCENT ?= 90

model-coverage-tmpdir:
	mkdir -p $(GOTMPDIR)

model-coverage: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),)

model-coverage-json: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -json

model-coverage-markdown: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -markdown

model-coverage-csv: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -csv

model-coverage-snapshot: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -snapshot

model-coverage-snapshot-file: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -snapshot > docs/model-coverage-snapshot.md

model-coverage-snapshot-check: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage -snapshot > $(GOTMPDIR)/model-coverage-snapshot.check.md
	cmp docs/model-coverage-snapshot.md $(GOTMPDIR)/model-coverage-snapshot.check.md

model-coverage-runtime-roadmap: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-roadmap

model-coverage-runtime-roadmap-json: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-roadmap-json

model-coverage-next-runtime: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -next-runtime

model-coverage-next-runtime-json: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -next-runtime-json

model-coverage-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -pending-only

model-coverage-references-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -references-only -pending-only

model-coverage-runtime-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-only -pending-only

model-coverage-execution-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -execution-only -pending-only

model-coverage-parity-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -parity-only -pending-only

model-coverage-readiness-pending: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -readiness-only -pending-only

model-coverage-references-gate: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -references-only -fail-pending

model-coverage-runtime-gate: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-only -fail-pending

model-coverage-execution-gate: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -execution-only -fail-pending

model-coverage-parity-gate: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -parity-only -fail-pending

model-coverage-readiness-gate: model-coverage-tmpdir
	go run ./cmd/models/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -readiness-only -fail-pending

vet:
	go vet ./...

clean:
	rm -rf bin/

models-list:
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) --dry-run $(MODEL_DOWNLOAD_FLAGS)

models-download:
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) $(MODEL_DOWNLOAD_FLAGS)

models-download-small:
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) --group small $(MODEL_DOWNLOAD_FLAGS)

models-download-qwen:
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) --group qwen $(MODEL_DOWNLOAD_FLAGS)

models-download-qwen3tts:
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) --group qwen3tts $(MODEL_DOWNLOAD_FLAGS)

models-download-lfm2:
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) --group lfm2 $(MODEL_DOWNLOAD_FLAGS)

models-download-gemma4:
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) --group gemma4 $(MODEL_DOWNLOAD_FLAGS)

models-download-speaker:
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) --group speaker $(MODEL_DOWNLOAD_FLAGS)

models-download-one:
	@if [ -z "$(MODEL)" ]; then echo "usage: make models-download-one MODEL=qwen3.6-27b-mlx4-mtp"; exit 2; fi
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) --only $(MODEL) $(MODEL_DOWNLOAD_FLAGS)

GGUF_MODEL ?= /opt/models/Qwen3.6-28B-REAP20-A3B-Q4_K_M.gguf
GGUF_PROMPT_IDS ?= 0
GGUF_MAX_NEW ?= 1
GGUF_CACHE_TYPE_K ?= turbo4
GGUF_CACHE_TYPE_V ?= turbo2
GGUF_KV_RESIDUAL_WINDOW ?= 128
GGUF_KV_SMOKE_TOKENS ?= 5
GGUF_EXPECT_KV_SMOKE_LAYER ?=
GGUF_EXPECT_KV_SMOKE_COMPRESSED ?=
GGUF_EXPECT_KV_SMOKE_FULL ?=
GGUF_EXPECT_KV_SMOKE_BYTES ?=
GGUF_EXPECT_KV_SMOKE_SCRATCH_BYTES ?=
GGUF_EXPECT_KV_SMOKE_TOTAL_BYTES ?=
GGUF_EXPECT_GENERATED ?=
GGUF_EXPECT_DECODED ?=
GGUF_EXPECT_RUNTIME_FLOAT_BYTES ?=
GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES ?=
GGUF_EXPECT_RUNTIME_SCRATCH_BYTES ?=
GGUF_EXPECT_RUNTIME_TOTAL_BYTES ?=
GGUF_EXPECT_KV_COMPRESSED_LAYERS ?=
GGUF_EXPECT_KV_SEQ ?=
GGUF_EXPECT_KV_COMPRESSED_COUNT ?=
GGUF_EXPECT_KV_FULL_COUNT ?=
GGUF_EXPECT_KV_FLOAT_BYTES ?=
GGUF_EXPECT_KV_COMPRESSED_BYTES ?=
GGUF_EXPECT_KV_SCRATCH_BYTES ?=
GGUF_EXPECT_KV_TOTAL_BYTES ?=
GGUF_EXPECT_REAP_RATIO ?=
GGUF_EXPECT_REAP_SOURCE ?=
GGUF_EXPECT_ARCHITECTURE ?=
GGUF_EXPECT_NAME_CONTAINS ?=
GGUF_EXPECT_TENSOR_COUNT ?=
GGUF_EXPECT_LAYERS ?=
GGUF_EXPECT_HIDDEN_SIZE ?=
GGUF_EXPECT_HEADS ?=
GGUF_EXPECT_VOCAB_SIZE ?=
GGUF_EXPECT_TOKENIZER_TOKENS ?=
GGUF_EXPECT_BOS ?=
GGUF_EXPECT_EOS ?=
GGUF_EXPECT_MAX_SEQ_LEN ?=
GGUF_EXPECT_FULL_ATTENTION_INTERVAL ?=
GGUF_EXPECT_KV_HEADS ?=
GGUF_EXPECT_HEAD_DIM ?=
GGUF_EXPECT_KV_DIM ?=
GGUF_EXPECT_EXPERTS ?=
GGUF_EXPECT_EXPERTS_PER_TOKEN ?=
GGUF_EXPECT_F32_COUNT ?=
GGUF_EXPECT_Q4_K_COUNT ?=
GGUF_EXPECT_Q6_K_COUNT ?=
GGUF_EXPECT_CACHE_LAYERS ?=
GGUF_EXPECT_PROTECTED_CACHE_LAYERS ?=
GGUF_EXPECT_FULL_KV_BYTES ?=
GGUF_EXPECT_ESTIMATED_KV_BYTES ?=
GGUF_EXPECT_SAVED_KV_BYTES ?=
GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES ?=
GGUF_EXPECT_ESTIMATED_TOTAL_BYTES ?=
GGUF_EXPECT_SIMD_ROTATION ?=
GGUF_CI_PACKAGES ?= ./cmd/llm/llmserver ./loader/gguf ./cmd/models/ggufinspect ./cmd/models/ggufsmoke ./model ./runtime/kv

# Inspect and smoke the native pure-Go/SIMD GGUF path for llama/Qwen REAP models.
gguf-inspect:
	go run ./cmd/models/ggufinspect -json -require-runtime-ready -cache-type-k $(GGUF_CACHE_TYPE_K) -cache-type-v $(GGUF_CACHE_TYPE_V) -kv-residual-window $(GGUF_KV_RESIDUAL_WINDOW) $(if $(GGUF_EXPECT_REAP_RATIO),-expect-reap-ratio $(GGUF_EXPECT_REAP_RATIO),) $(if $(GGUF_EXPECT_REAP_SOURCE),-expect-reap-source $(GGUF_EXPECT_REAP_SOURCE),) $(if $(GGUF_EXPECT_ARCHITECTURE),-expect-architecture $(GGUF_EXPECT_ARCHITECTURE),) $(if $(GGUF_EXPECT_NAME_CONTAINS),-expect-name-contains $(GGUF_EXPECT_NAME_CONTAINS),) $(if $(GGUF_EXPECT_TENSOR_COUNT),-expect-tensor-count $(GGUF_EXPECT_TENSOR_COUNT),) $(if $(GGUF_EXPECT_LAYERS),-expect-layers $(GGUF_EXPECT_LAYERS),) $(if $(GGUF_EXPECT_HIDDEN_SIZE),-expect-hidden-size $(GGUF_EXPECT_HIDDEN_SIZE),) $(if $(GGUF_EXPECT_HEADS),-expect-heads $(GGUF_EXPECT_HEADS),) $(if $(GGUF_EXPECT_VOCAB_SIZE),-expect-vocab-size $(GGUF_EXPECT_VOCAB_SIZE),) $(if $(GGUF_EXPECT_TOKENIZER_TOKENS),-expect-tokenizer-tokens $(GGUF_EXPECT_TOKENIZER_TOKENS),) $(if $(GGUF_EXPECT_BOS),-expect-bos $(GGUF_EXPECT_BOS),) $(if $(GGUF_EXPECT_EOS),-expect-eos $(GGUF_EXPECT_EOS),) $(if $(GGUF_EXPECT_MAX_SEQ_LEN),-expect-max-seq-len $(GGUF_EXPECT_MAX_SEQ_LEN),) $(if $(GGUF_EXPECT_FULL_ATTENTION_INTERVAL),-expect-full-attention-interval $(GGUF_EXPECT_FULL_ATTENTION_INTERVAL),) $(if $(GGUF_EXPECT_KV_HEADS),-expect-kv-heads $(GGUF_EXPECT_KV_HEADS),) $(if $(GGUF_EXPECT_HEAD_DIM),-expect-head-dim $(GGUF_EXPECT_HEAD_DIM),) $(if $(GGUF_EXPECT_KV_DIM),-expect-kv-dim $(GGUF_EXPECT_KV_DIM),) $(if $(GGUF_EXPECT_EXPERTS),-expect-experts $(GGUF_EXPECT_EXPERTS),) $(if $(GGUF_EXPECT_EXPERTS_PER_TOKEN),-expect-experts-per-token $(GGUF_EXPECT_EXPERTS_PER_TOKEN),) $(if $(GGUF_EXPECT_F32_COUNT),-expect-f32-count $(GGUF_EXPECT_F32_COUNT),) $(if $(GGUF_EXPECT_Q4_K_COUNT),-expect-q4-k-count $(GGUF_EXPECT_Q4_K_COUNT),) $(if $(GGUF_EXPECT_Q6_K_COUNT),-expect-q6-k-count $(GGUF_EXPECT_Q6_K_COUNT),) $(if $(GGUF_EXPECT_CACHE_LAYERS),-expect-cache-layers $(GGUF_EXPECT_CACHE_LAYERS),) $(if $(GGUF_EXPECT_PROTECTED_CACHE_LAYERS),-expect-protected-cache-layers $(GGUF_EXPECT_PROTECTED_CACHE_LAYERS),) $(if $(GGUF_EXPECT_FULL_KV_BYTES),-expect-full-kv-bytes $(GGUF_EXPECT_FULL_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_KV_BYTES),-expect-estimated-kv-bytes $(GGUF_EXPECT_ESTIMATED_KV_BYTES),) $(if $(GGUF_EXPECT_SAVED_KV_BYTES),-expect-saved-kv-bytes $(GGUF_EXPECT_SAVED_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),-expect-estimated-scratch-bytes $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),-expect-estimated-total-bytes $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),) $(if $(GGUF_EXPECT_SIMD_ROTATION),-expect-simd-rotation,) $(GGUF_MODEL)

gguf-smoke:
	go run ./cmd/models/ggufsmoke -model $(GGUF_MODEL) -prompt-ids $(GGUF_PROMPT_IDS) -max-new $(GGUF_MAX_NEW) -cache-type-k $(GGUF_CACHE_TYPE_K) -cache-type-v $(GGUF_CACHE_TYPE_V) -kv-residual-window $(GGUF_KV_RESIDUAL_WINDOW) $(if $(GGUF_EXPECT_FULL_KV_BYTES),-expect-full-kv-bytes $(GGUF_EXPECT_FULL_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_KV_BYTES),-expect-estimated-kv-bytes $(GGUF_EXPECT_ESTIMATED_KV_BYTES),) $(if $(GGUF_EXPECT_SAVED_KV_BYTES),-expect-saved-kv-bytes $(GGUF_EXPECT_SAVED_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),-expect-estimated-scratch-bytes $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),-expect-estimated-total-bytes $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),) $(if $(GGUF_EXPECT_GENERATED),-expect-generated $(GGUF_EXPECT_GENERATED),) $(if $(GGUF_EXPECT_DECODED),-expect-decoded $(GGUF_EXPECT_DECODED),) $(if $(GGUF_EXPECT_RUNTIME_FLOAT_BYTES),-expect-runtime-float-bytes $(GGUF_EXPECT_RUNTIME_FLOAT_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES),-expect-runtime-compressed-bytes $(GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_SCRATCH_BYTES),-expect-runtime-scratch-bytes $(GGUF_EXPECT_RUNTIME_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_TOTAL_BYTES),-expect-runtime-total-bytes $(GGUF_EXPECT_RUNTIME_TOTAL_BYTES),) $(if $(GGUF_EXPECT_SIMD_ROTATION),-expect-simd-rotation,)

gguf-bench:
	go run ./cmd/models/ggufsmoke -model $(GGUF_MODEL) -prompt-ids $(GGUF_PROMPT_IDS) -max-new $(GGUF_MAX_NEW) -cache-type-k $(GGUF_CACHE_TYPE_K) -cache-type-v $(GGUF_CACHE_TYPE_V) -kv-residual-window $(GGUF_KV_RESIDUAL_WINDOW) $(if $(GGUF_EXPECT_FULL_KV_BYTES),-expect-full-kv-bytes $(GGUF_EXPECT_FULL_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_KV_BYTES),-expect-estimated-kv-bytes $(GGUF_EXPECT_ESTIMATED_KV_BYTES),) $(if $(GGUF_EXPECT_SAVED_KV_BYTES),-expect-saved-kv-bytes $(GGUF_EXPECT_SAVED_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),-expect-estimated-scratch-bytes $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),-expect-estimated-total-bytes $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),) $(if $(GGUF_EXPECT_GENERATED),-expect-generated $(GGUF_EXPECT_GENERATED),) $(if $(GGUF_EXPECT_DECODED),-expect-decoded $(GGUF_EXPECT_DECODED),) $(if $(GGUF_EXPECT_RUNTIME_FLOAT_BYTES),-expect-runtime-float-bytes $(GGUF_EXPECT_RUNTIME_FLOAT_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES),-expect-runtime-compressed-bytes $(GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_SCRATCH_BYTES),-expect-runtime-scratch-bytes $(GGUF_EXPECT_RUNTIME_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_RUNTIME_TOTAL_BYTES),-expect-runtime-total-bytes $(GGUF_EXPECT_RUNTIME_TOTAL_BYTES),) $(if $(GGUF_EXPECT_KV_COMPRESSED_LAYERS),-expect-kv-compressed-layers $(GGUF_EXPECT_KV_COMPRESSED_LAYERS),) $(if $(GGUF_EXPECT_KV_SEQ),-expect-kv-seq $(GGUF_EXPECT_KV_SEQ),) $(if $(GGUF_EXPECT_KV_COMPRESSED_COUNT),-expect-kv-compressed-count $(GGUF_EXPECT_KV_COMPRESSED_COUNT),) $(if $(GGUF_EXPECT_KV_FULL_COUNT),-expect-kv-full-count $(GGUF_EXPECT_KV_FULL_COUNT),) $(if $(GGUF_EXPECT_KV_FLOAT_BYTES),-expect-kv-float-bytes $(GGUF_EXPECT_KV_FLOAT_BYTES),) $(if $(GGUF_EXPECT_KV_COMPRESSED_BYTES),-expect-kv-compressed-bytes $(GGUF_EXPECT_KV_COMPRESSED_BYTES),) $(if $(GGUF_EXPECT_KV_SCRATCH_BYTES),-expect-kv-scratch-bytes $(GGUF_EXPECT_KV_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_KV_TOTAL_BYTES),-expect-kv-total-bytes $(GGUF_EXPECT_KV_TOTAL_BYTES),) $(if $(GGUF_EXPECT_SIMD_ROTATION),-expect-simd-rotation,) -bench

gguf-turboquant-smoke:
	go run ./cmd/models/ggufsmoke -model $(GGUF_MODEL) -load-only -cache-type-k $(GGUF_CACHE_TYPE_K) -cache-type-v $(GGUF_CACHE_TYPE_V) -kv-residual-window $(GGUF_KV_RESIDUAL_WINDOW) -kv-smoke-tokens $(GGUF_KV_SMOKE_TOKENS) $(if $(GGUF_EXPECT_FULL_KV_BYTES),-expect-full-kv-bytes $(GGUF_EXPECT_FULL_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_KV_BYTES),-expect-estimated-kv-bytes $(GGUF_EXPECT_ESTIMATED_KV_BYTES),) $(if $(GGUF_EXPECT_SAVED_KV_BYTES),-expect-saved-kv-bytes $(GGUF_EXPECT_SAVED_KV_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),-expect-estimated-scratch-bytes $(GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),-expect-estimated-total-bytes $(GGUF_EXPECT_ESTIMATED_TOTAL_BYTES),) $(if $(GGUF_EXPECT_KV_SMOKE_LAYER),-expect-kv-smoke-layer $(GGUF_EXPECT_KV_SMOKE_LAYER),) $(if $(GGUF_EXPECT_KV_SMOKE_COMPRESSED),-expect-kv-smoke-compressed $(GGUF_EXPECT_KV_SMOKE_COMPRESSED),) $(if $(GGUF_EXPECT_KV_SMOKE_FULL),-expect-kv-smoke-full $(GGUF_EXPECT_KV_SMOKE_FULL),) $(if $(GGUF_EXPECT_KV_SMOKE_BYTES),-expect-kv-smoke-bytes $(GGUF_EXPECT_KV_SMOKE_BYTES),) $(if $(GGUF_EXPECT_KV_SMOKE_SCRATCH_BYTES),-expect-kv-smoke-scratch-bytes $(GGUF_EXPECT_KV_SMOKE_SCRATCH_BYTES),) $(if $(GGUF_EXPECT_KV_SMOKE_TOTAL_BYTES),-expect-kv-smoke-total-bytes $(GGUF_EXPECT_KV_SMOKE_TOTAL_BYTES),) $(if $(GGUF_EXPECT_SIMD_ROTATION),-expect-simd-rotation,)

gguf-validate: gguf-inspect gguf-smoke gguf-turboquant-smoke

gguf-check: gguf-validate gguf-bench

gguf-ci:
	go test $(GGUF_CI_PACKAGES) -run '^$$'
	$(MAKE) gguf-validate

gguf-inspect-qwen36-reap:
	$(MAKE) gguf-inspect GGUF_MODEL=$(GGUF_MODEL) GGUF_CACHE_TYPE_K=turbo4 GGUF_CACHE_TYPE_V=turbo2 GGUF_KV_RESIDUAL_WINDOW=2 GGUF_EXPECT_REAP_RATIO=0.20 GGUF_EXPECT_REAP_SOURCE=filename_or_name GGUF_EXPECT_ARCHITECTURE=qwen35moe GGUF_EXPECT_NAME_CONTAINS=REAP20 GGUF_EXPECT_TENSOR_COUNT=733 GGUF_EXPECT_LAYERS=40 GGUF_EXPECT_HIDDEN_SIZE=2048 GGUF_EXPECT_HEADS=16 GGUF_EXPECT_VOCAB_SIZE=248320 GGUF_EXPECT_TOKENIZER_TOKENS=248320 GGUF_EXPECT_BOS=248044 GGUF_EXPECT_EOS=248046 GGUF_EXPECT_MAX_SEQ_LEN=262144 GGUF_EXPECT_FULL_ATTENTION_INTERVAL=4 GGUF_EXPECT_KV_HEADS=2 GGUF_EXPECT_HEAD_DIM=256 GGUF_EXPECT_KV_DIM=512 GGUF_EXPECT_EXPERTS=205 GGUF_EXPECT_EXPERTS_PER_TOKEN=8 GGUF_EXPECT_F32_COUNT=301 GGUF_EXPECT_Q4_K_COUNT=371 GGUF_EXPECT_Q6_K_COUNT=61 GGUF_EXPECT_CACHE_LAYERS=10 GGUF_EXPECT_PROTECTED_CACHE_LAYERS=1 GGUF_EXPECT_FULL_KV_BYTES=10737418240 GGUF_EXPECT_ESTIMATED_KV_BYTES=2055275200 GGUF_EXPECT_SAVED_KV_BYTES=8682143040 GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES=9663699456 GGUF_EXPECT_ESTIMATED_TOTAL_BYTES=11718974656 GGUF_EXPECT_SIMD_ROTATION=1

gguf-smoke-qwen36-reap:
	$(MAKE) gguf-smoke GGUF_MODEL=$(GGUF_MODEL) GGUF_PROMPT_IDS=0 GGUF_MAX_NEW=1 GGUF_CACHE_TYPE_K=turbo4 GGUF_CACHE_TYPE_V=turbo2 GGUF_KV_RESIDUAL_WINDOW=2 GGUF_EXPECT_GENERATED=489 GGUF_EXPECT_DECODED=ype GGUF_EXPECT_RUNTIME_FLOAT_BYTES=245760 GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES=81920 GGUF_EXPECT_RUNTIME_SCRATCH_BYTES=96768 GGUF_EXPECT_RUNTIME_TOTAL_BYTES=424448 GGUF_EXPECT_SIMD_ROTATION=1

gguf-validate-qwen36-reap:
	$(MAKE) gguf-validate GGUF_MODEL=$(GGUF_MODEL) GGUF_PROMPT_IDS=0 GGUF_MAX_NEW=1 GGUF_CACHE_TYPE_K=turbo4 GGUF_CACHE_TYPE_V=turbo2 GGUF_KV_RESIDUAL_WINDOW=2 GGUF_EXPECT_REAP_RATIO=0.20 GGUF_EXPECT_REAP_SOURCE=filename_or_name GGUF_EXPECT_ARCHITECTURE=qwen35moe GGUF_EXPECT_NAME_CONTAINS=REAP20 GGUF_EXPECT_TENSOR_COUNT=733 GGUF_EXPECT_LAYERS=40 GGUF_EXPECT_HIDDEN_SIZE=2048 GGUF_EXPECT_HEADS=16 GGUF_EXPECT_VOCAB_SIZE=248320 GGUF_EXPECT_TOKENIZER_TOKENS=248320 GGUF_EXPECT_BOS=248044 GGUF_EXPECT_EOS=248046 GGUF_EXPECT_MAX_SEQ_LEN=262144 GGUF_EXPECT_FULL_ATTENTION_INTERVAL=4 GGUF_EXPECT_KV_HEADS=2 GGUF_EXPECT_HEAD_DIM=256 GGUF_EXPECT_KV_DIM=512 GGUF_EXPECT_EXPERTS=205 GGUF_EXPECT_EXPERTS_PER_TOKEN=8 GGUF_EXPECT_F32_COUNT=301 GGUF_EXPECT_Q4_K_COUNT=371 GGUF_EXPECT_Q6_K_COUNT=61 GGUF_EXPECT_CACHE_LAYERS=10 GGUF_EXPECT_PROTECTED_CACHE_LAYERS=1 GGUF_EXPECT_FULL_KV_BYTES=10737418240 GGUF_EXPECT_ESTIMATED_KV_BYTES=2055275200 GGUF_EXPECT_SAVED_KV_BYTES=8682143040 GGUF_EXPECT_ESTIMATED_SCRATCH_BYTES=9663699456 GGUF_EXPECT_ESTIMATED_TOTAL_BYTES=11718974656 GGUF_EXPECT_KV_SMOKE_LAYER=3 GGUF_EXPECT_KV_SMOKE_COMPRESSED=3 GGUF_EXPECT_KV_SMOKE_FULL=2 GGUF_EXPECT_KV_SMOKE_BYTES=9440 GGUF_EXPECT_KV_SMOKE_SCRATCH_BYTES=1280 GGUF_EXPECT_KV_SMOKE_TOTAL_BYTES=10720 GGUF_EXPECT_GENERATED=489 GGUF_EXPECT_DECODED=ype GGUF_EXPECT_RUNTIME_FLOAT_BYTES=245760 GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES=81920 GGUF_EXPECT_RUNTIME_SCRATCH_BYTES=96768 GGUF_EXPECT_RUNTIME_TOTAL_BYTES=424448 GGUF_EXPECT_SIMD_ROTATION=1

gguf-bench-qwen36-reap:
	$(MAKE) gguf-bench GGUF_MODEL=$(GGUF_MODEL) GGUF_PROMPT_IDS=0 GGUF_MAX_NEW=1 GGUF_CACHE_TYPE_K=turbo4 GGUF_CACHE_TYPE_V=turbo2 GGUF_KV_RESIDUAL_WINDOW=2 GGUF_EXPECT_GENERATED=489 GGUF_EXPECT_DECODED=ype GGUF_EXPECT_RUNTIME_FLOAT_BYTES=245760 GGUF_EXPECT_RUNTIME_COMPRESSED_BYTES=81920 GGUF_EXPECT_RUNTIME_SCRATCH_BYTES=96768 GGUF_EXPECT_RUNTIME_TOTAL_BYTES=424448 GGUF_EXPECT_KV_COMPRESSED_LAYERS=10 GGUF_EXPECT_KV_SEQ=2 GGUF_EXPECT_KV_COMPRESSED_COUNT=0 GGUF_EXPECT_KV_FULL_COUNT=20 GGUF_EXPECT_KV_FLOAT_BYTES=245760 GGUF_EXPECT_KV_COMPRESSED_BYTES=81920 GGUF_EXPECT_KV_SCRATCH_BYTES=0 GGUF_EXPECT_KV_TOTAL_BYTES=327680 GGUF_EXPECT_SIMD_ROTATION=1

gguf-check-qwen36-reap: gguf-validate-qwen36-reap gguf-bench-qwen36-reap

gguf-ci-qwen36-reap:
	go test $(GGUF_CI_PACKAGES) -run '^$$'
	$(MAKE) gguf-check-qwen36-reap

QWEN3TTS_MODEL ?=
QWEN3TTS_TEXT ?= Hello world
QWEN3TTS_SPEAKER ?= ryan
QWEN3TTS_LANGUAGE ?= en
QWEN3TTS_INSPECT_FLAGS ?= -json
QWEN3TTS_FIXTURE ?= model/qwen3tts/testdata/customvoice_prompt_fixture.json
QWEN3TTS_FIXTURE_FLAGS ?= -json
LFM2_MODEL ?=
LFM2_INSPECT_FLAGS ?= -json
LFM2_FIXTURE ?= model/lfm2/testdata/lfm25_8b_a1b_metadata.json
LFM2_FIXTURE_FLAGS ?= -json

qwen3tts-inspect:
	@if [ -z "$(QWEN3TTS_MODEL)" ]; then echo "usage: make qwen3tts-inspect QWEN3TTS_MODEL=models/qwen3-tts-0.6b-customvoice [QWEN3TTS_TEXT='Hello world']"; exit 2; fi
	go run ./cmd/qwen/qwen3ttsinspect -model $(QWEN3TTS_MODEL) -text "$(QWEN3TTS_TEXT)" -speaker $(QWEN3TTS_SPEAKER) -language $(QWEN3TTS_LANGUAGE) $(QWEN3TTS_INSPECT_FLAGS)

qwen3tts-fixture-coverage:
	@if [ -z "$(QWEN3TTS_MODEL)" ]; then echo "usage: make qwen3tts-fixture-coverage QWEN3TTS_MODEL=models/qwen3-tts-0.6b-customvoice [QWEN3TTS_FIXTURE=model/qwen3tts/testdata/customvoice_prompt_fixture.json]"; exit 2; fi
	go run ./cmd/qwen/qwen3ttsinspect -model $(QWEN3TTS_MODEL) -fixture $(QWEN3TTS_FIXTURE) $(QWEN3TTS_FIXTURE_FLAGS)

lfm2-inspect:
	@if [ -z "$(LFM2_MODEL)" ]; then echo "usage: make lfm2-inspect LFM2_MODEL=models/lfm2.5-8b-a1b"; exit 2; fi
	go run ./cmd/models/lfm2inspect -model $(LFM2_MODEL) $(LFM2_INSPECT_FLAGS)

lfm2-fixture-coverage:
	@if [ -z "$(LFM2_MODEL)" ]; then echo "usage: make lfm2-fixture-coverage LFM2_MODEL=models/lfm2.5-8b-a1b [LFM2_FIXTURE=model/lfm2/testdata/lfm25_8b_a1b_metadata.json]"; exit 2; fi
	go run ./cmd/models/lfm2inspect -model $(LFM2_MODEL) -fixture $(LFM2_FIXTURE) $(LFM2_FIXTURE_FLAGS)

HUNYUAN3D_REPO ?= tencent/Hunyuan3D-2mini
HUNYUAN3D_SUBFOLDER ?= hunyuan3d-dit-v2-mini
HUNYUAN3D_INVENTORY ?= /workspace/tmp/hunyuan3d-mini-inventory.json
HUNYUAN3D_INVENTORY_FLAGS ?= --include-tensors
HUNYUAN3D_IMAGE_FIXTURE ?= /workspace/tmp/hunyuan3d-image-preprocess-fixture.json
HUNYUAN3D_IMAGE ?=
HUNYUAN3D_IMAGE_FLAGS ?=
HUNYUAN3D_SEAHORSE_IMAGE ?= testdata/hunyuan3d/seahorse_rgba.png
HUNYUAN3D_SEAHORSE_OUT ?= /workspace/tmp/hunyuan3d-seahorse.glb
HUNYUAN3D_SEAHORSE_FLAGS ?=
HUNYUAN3D_SRC ?= /workspace/tmp/Hunyuan3D-2-info
HUNYUAN3D_CONFIG ?=
HUNYUAN3D_CHECKPOINT ?=
HUNYUAN3D_INSPECT_FLAGS ?=
HUNYUAN3D_ENV_REPORT ?= /workspace/tmp/hunyuan3d-fixture-env.json
HUNYUAN3D_CONDITIONER_FIXTURE ?= /workspace/tmp/hunyuan3d-conditioner-fixture.json
HUNYUAN3D_CONDITIONER_FLAGS ?=
HUNYUAN3D_DENOISER_FIXTURE ?= /workspace/tmp/hunyuan3d-denoiser-step-fixture.json
HUNYUAN3D_DENOISER_FLAGS ?=
HUNYUAN3D_LOWSTEP_FIXTURE ?= /workspace/tmp/hunyuan3d-lowstep-latents-fixture.json
HUNYUAN3D_LOWSTEP_FLAGS ?=
HUNYUAN3D_MESH_FIXTURE ?= /workspace/tmp/hunyuan3d-mesh-fixture.json
HUNYUAN3D_MESH_FLAGS ?=
TRELLIS2_REPO ?= microsoft/TRELLIS.2-4B
TRELLIS2_REVISION ?= main
TRELLIS2_LOCAL_DIR ?=
TRELLIS2_INVENTORY ?= /workspace/tmp/trellis2-inventory.json
TRELLIS2_INVENTORY_FLAGS ?=
TRELLIS2_SRC ?= /workspace/tmp/TRELLIS.2
TRELLIS2_MODEL_DIR ?= microsoft/TRELLIS.2-4B
TRELLIS2_IMAGE ?=
TRELLIS2_ENV_REPORT ?= /workspace/tmp/trellis2-fixture-env.json
TRELLIS2_LOWSTEP_FIXTURE ?= /workspace/tmp/trellis2-lowstep-fixture.json
TRELLIS2_LOWSTEP_FLAGS ?=
TRELLIS2_OVOXEL_FILES ?=
TRELLIS2_OVOXEL_INSPECT ?= /workspace/tmp/trellis2-ovoxel-inspect.json

hunyuan3d-fixture-env:
	$(PYTHON) scripts/hunyuan3d_check_fixture_env.py --hunyuan3d-src $(HUNYUAN3D_SRC) $(if $(HUNYUAN3D_CONFIG),--config $(HUNYUAN3D_CONFIG),) $(if $(HUNYUAN3D_CHECKPOINT),--checkpoint $(HUNYUAN3D_CHECKPOINT),) $(if $(HUNYUAN3D_IMAGE),--image $(HUNYUAN3D_IMAGE),) --out $(HUNYUAN3D_ENV_REPORT)

hunyuan3d-inventory:
	$(PYTHON) scripts/hunyuan3d_fixture_inventory.py --repo $(HUNYUAN3D_REPO) --subfolder $(HUNYUAN3D_SUBFOLDER) --out $(HUNYUAN3D_INVENTORY) $(HUNYUAN3D_INVENTORY_FLAGS)

trellis2-fixture-env:
	$(PYTHON) scripts/trellis2_check_fixture_env.py --trellis2-src $(TRELLIS2_SRC) --model-dir $(TRELLIS2_MODEL_DIR) $(if $(TRELLIS2_IMAGE),--image $(TRELLIS2_IMAGE),) --out $(TRELLIS2_ENV_REPORT)

trellis2-inventory:
	$(PYTHON) scripts/trellis2_fixture_inventory.py --repo $(TRELLIS2_REPO) --revision $(TRELLIS2_REVISION) --out $(TRELLIS2_INVENTORY) $(if $(TRELLIS2_LOCAL_DIR),--local-dir $(TRELLIS2_LOCAL_DIR),) $(TRELLIS2_INVENTORY_FLAGS)

trellis2-lowstep-fixture:
	@if [ -z "$(TRELLIS2_IMAGE)" ]; then echo "usage: make trellis2-lowstep-fixture TRELLIS2_IMAGE=...png [TRELLIS2_SRC=/path/to/TRELLIS.2] [TRELLIS2_MODEL_DIR=/path/or/hf-id]"; exit 2; fi
	$(PYTHON) scripts/trellis2_lowstep_fixture.py --trellis2-src $(TRELLIS2_SRC) --model-dir $(TRELLIS2_MODEL_DIR) --image $(TRELLIS2_IMAGE) --out $(TRELLIS2_LOWSTEP_FIXTURE) $(TRELLIS2_LOWSTEP_FLAGS)

trellis2-ovoxel-inspect:
	@if [ -z "$(TRELLIS2_OVOXEL_FILES)" ]; then echo "usage: make trellis2-ovoxel-inspect TRELLIS2_OVOXEL_FILES='file1.npz file2.vxz'"; exit 2; fi
	$(PYTHON) scripts/trellis2_ovoxel_inspect.py --out $(TRELLIS2_OVOXEL_INSPECT) $(TRELLIS2_OVOXEL_FILES)

hunyuan3d-inspect:
	@if [ -z "$(HUNYUAN3D_CONFIG)" ]; then echo "usage: make hunyuan3d-inspect HUNYUAN3D_CONFIG=.../config.yaml [HUNYUAN3D_CHECKPOINT=.../model.safetensors]"; exit 2; fi
	go run ./cmd/image/hy3dinspect -config $(HUNYUAN3D_CONFIG) $(if $(HUNYUAN3D_CHECKPOINT),-safetensors $(HUNYUAN3D_CHECKPOINT),) $(HUNYUAN3D_INSPECT_FLAGS)

hunyuan3d-seahorse:
	$(PYTHON) scripts/hunyuan3d_seahorse_demo.py --hunyuan3d-src $(HUNYUAN3D_SRC) --image $(HUNYUAN3D_SEAHORSE_IMAGE) --out $(HUNYUAN3D_SEAHORSE_OUT) --model $(HUNYUAN3D_REPO) --subfolder $(HUNYUAN3D_SUBFOLDER) $(HUNYUAN3D_SEAHORSE_FLAGS)

hunyuan3d-image-fixture:
	$(PYTHON) scripts/hunyuan3d_image_fixture.py --out $(HUNYUAN3D_IMAGE_FIXTURE) $(if $(HUNYUAN3D_IMAGE),--image $(HUNYUAN3D_IMAGE),) $(HUNYUAN3D_IMAGE_FLAGS)

hunyuan3d-conditioner-fixture:
	@if [ -z "$(HUNYUAN3D_CONFIG)" ] || [ -z "$(HUNYUAN3D_CHECKPOINT)" ] || [ -z "$(HUNYUAN3D_IMAGE)" ]; then echo "usage: make hunyuan3d-conditioner-fixture HUNYUAN3D_CONFIG=.../config.yaml HUNYUAN3D_CHECKPOINT=.../model.fp16.safetensors HUNYUAN3D_IMAGE=...png"; exit 2; fi
	$(PYTHON) scripts/hunyuan3d_conditioner_fixture.py --hunyuan3d-src $(HUNYUAN3D_SRC) --config $(HUNYUAN3D_CONFIG) --checkpoint $(HUNYUAN3D_CHECKPOINT) --image $(HUNYUAN3D_IMAGE) --out $(HUNYUAN3D_CONDITIONER_FIXTURE) $(HUNYUAN3D_CONDITIONER_FLAGS)

hunyuan3d-denoiser-fixture:
	@if [ -z "$(HUNYUAN3D_CONFIG)" ] || [ -z "$(HUNYUAN3D_CHECKPOINT)" ] || [ -z "$(HUNYUAN3D_IMAGE)" ]; then echo "usage: make hunyuan3d-denoiser-fixture HUNYUAN3D_CONFIG=.../config.yaml HUNYUAN3D_CHECKPOINT=.../model.fp16.safetensors HUNYUAN3D_IMAGE=...png"; exit 2; fi
	$(PYTHON) scripts/hunyuan3d_denoiser_fixture.py --hunyuan3d-src $(HUNYUAN3D_SRC) --config $(HUNYUAN3D_CONFIG) --checkpoint $(HUNYUAN3D_CHECKPOINT) --image $(HUNYUAN3D_IMAGE) --out $(HUNYUAN3D_DENOISER_FIXTURE) $(HUNYUAN3D_DENOISER_FLAGS)

hunyuan3d-lowstep-fixture:
	@if [ -z "$(HUNYUAN3D_CONFIG)" ] || [ -z "$(HUNYUAN3D_CHECKPOINT)" ] || [ -z "$(HUNYUAN3D_IMAGE)" ]; then echo "usage: make hunyuan3d-lowstep-fixture HUNYUAN3D_CONFIG=.../config.yaml HUNYUAN3D_CHECKPOINT=.../model.fp16.safetensors HUNYUAN3D_IMAGE=...png"; exit 2; fi
	$(PYTHON) scripts/hunyuan3d_lowstep_latent_fixture.py --hunyuan3d-src $(HUNYUAN3D_SRC) --config $(HUNYUAN3D_CONFIG) --checkpoint $(HUNYUAN3D_CHECKPOINT) --image $(HUNYUAN3D_IMAGE) --out $(HUNYUAN3D_LOWSTEP_FIXTURE) $(HUNYUAN3D_LOWSTEP_FLAGS)

hunyuan3d-mesh-fixture:
	@if [ -z "$(HUNYUAN3D_CONFIG)" ] || [ -z "$(HUNYUAN3D_CHECKPOINT)" ] || [ -z "$(HUNYUAN3D_IMAGE)" ]; then echo "usage: make hunyuan3d-mesh-fixture HUNYUAN3D_CONFIG=.../config.yaml HUNYUAN3D_CHECKPOINT=.../model.fp16.safetensors HUNYUAN3D_IMAGE=...png"; exit 2; fi
	$(PYTHON) scripts/hunyuan3d_mesh_fixture.py --hunyuan3d-src $(HUNYUAN3D_SRC) --config $(HUNYUAN3D_CONFIG) --checkpoint $(HUNYUAN3D_CHECKPOINT) --image $(HUNYUAN3D_IMAGE) --out $(HUNYUAN3D_MESH_FIXTURE) $(HUNYUAN3D_MESH_FLAGS)

# GPU-heavy tests (require GEMMA4_TRACE_TEST=1 and GPU)
test-gpu:
	GEMMA4_TRACE_TEST=1 go test -tags diagnostic -count=1 -run TestGemma4GPUBench ./model -v

# Quick smoke test
smoke:
	@echo "=== build ==="
	go build -o /dev/null ./cmd/llm/llmgen
	go build -o /dev/null ./cmd/llm/llmserver
	go build -o /dev/null ./cmd/llm/llmchat
	@echo "=== vet ==="
	go vet ./...
	@echo "=== unit tests ==="
	go test -count=1 -timeout=60s ./loader/... ./backends/nvidia/... ./backends/placement/... ./backends/simd/... ./backends/vulkan/... ./runtime/... ./tensor/...
	@echo "=== ok ==="

# Ideogram 4 OSS/ComfyUI-style generation. ComfyUI-Ideogram4's recommended
# workflow is Magic Prompt -> Generate, where Magic Prompt produces a structured
# single-line JSON caption. Keep the prompt in a file and pass it verbatim so
# generation runs do not drift back to plain natural-language prompts.
IDEOGRAM4_MODEL ?= /srv/piclaw-dev/workspace/tmp/ideogram4-cat-model
IDEOGRAM4_PROMPT_FILE ?= prompts/ideogram4/cat.json
IDEOGRAM4_OUT ?= $(TMPDIR)/ideogram4/cat_comfy_prompt_256.png
IDEOGRAM4_WIDTH ?= 256
IDEOGRAM4_HEIGHT ?= 256
IDEOGRAM4_STEPS ?= 16
IDEOGRAM4_GUIDANCE ?= 7.0
IDEOGRAM4_MU ?= 0.0
IDEOGRAM4_STD ?= 1.75
IDEOGRAM4_SEED ?= 2026060803
IDEOGRAM4_GPU_RESIDENCY ?= phase
IDEOGRAM4_EXTRA_FLAGS ?= -gpu-fp8-sgemm
IDEOGRAM4_LAYER_CACHE_WINDOW ?= 21
IDEOGRAM4_LAYER_CACHE_START ?= 0
IDEOGRAM4_FULL_LAYER ?= 1
IDEOGRAM4_HIDDEN_RESIDENT ?= 1
IDEOGRAM4_LAYER_CACHE_ATTENTION_ALL ?= 0
IDEOGRAM4_LAYER_CACHE_WINDOW_COND ?= 34
IDEOGRAM4_LAYER_CACHE_WINDOW_UNCOND ?= 9
IDEOGRAM4_LAYER_CACHE_START_COND ?=
IDEOGRAM4_LAYER_CACHE_START_UNCOND ?=
IDEOGRAM4_LAYER_CACHE_QKV_ALL ?= 0
IDEOGRAM4_LAYER_CACHE_O_ALL ?= 0
IDEOGRAM4_ADALN_RESIDENT ?= 0
IDEOGRAM4_GPU_ENV ?= GO_PHERENCE_IDEOGRAM4_GPU_ADALN_RESIDENT=$(IDEOGRAM4_ADALN_RESIDENT) GO_PHERENCE_IDEOGRAM4_GPU_DIT_VECTOR=1 GO_PHERENCE_IDEOGRAM4_GPU_FULL_LAYER=$(IDEOGRAM4_FULL_LAYER) GO_PHERENCE_IDEOGRAM4_GPU_HIDDEN_RESIDENT=$(IDEOGRAM4_HIDDEN_RESIDENT) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW=$(IDEOGRAM4_LAYER_CACHE_WINDOW) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START=$(IDEOGRAM4_LAYER_CACHE_START) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW_COND=$(IDEOGRAM4_LAYER_CACHE_WINDOW_COND) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW_UNCOND=$(IDEOGRAM4_LAYER_CACHE_WINDOW_UNCOND) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START_COND=$(IDEOGRAM4_LAYER_CACHE_START_COND) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START_UNCOND=$(IDEOGRAM4_LAYER_CACHE_START_UNCOND) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_ATTENTION_ALL=$(IDEOGRAM4_LAYER_CACHE_ATTENTION_ALL) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_QKV_ALL=$(IDEOGRAM4_LAYER_CACHE_QKV_ALL) GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_O_ALL=$(IDEOGRAM4_LAYER_CACHE_O_ALL)

.PHONY: ideogram4-cat-prompt ideogram4-cat-gpu ideogram4-cat-cpu ideogram4-cat-open

ideogram4-cat-prompt:
	$(PYTHON) -m json.tool $(IDEOGRAM4_PROMPT_FILE) >/dev/null
	@cat $(IDEOGRAM4_PROMPT_FILE)
	@echo

ideogram4-cat-gpu: ideogram4-cat-prompt
	mkdir -p $(dir $(IDEOGRAM4_OUT)) $(GOTMPDIR)
	$(IDEOGRAM4_GPU_ENV) go run ./cmd/image/ideogram4gen \
		-model $(IDEOGRAM4_MODEL) \
		-prompt "$$(cat $(IDEOGRAM4_PROMPT_FILE))" \
		-out $(IDEOGRAM4_OUT) \
		-width $(IDEOGRAM4_WIDTH) \
		-height $(IDEOGRAM4_HEIGHT) \
		-steps $(IDEOGRAM4_STEPS) \
		-guidance $(IDEOGRAM4_GUIDANCE) \
		-mu $(IDEOGRAM4_MU) \
		-std $(IDEOGRAM4_STD) \
		-seed $(IDEOGRAM4_SEED) \
		-gpu -gpu-fp8 -gpu-fp8-cache -gpu-residency $(IDEOGRAM4_GPU_RESIDENCY) \
		$(IDEOGRAM4_EXTRA_FLAGS) \
		-timing

ideogram4-cat-cpu: ideogram4-cat-prompt
	mkdir -p $(dir $(IDEOGRAM4_OUT)) $(GOTMPDIR)
	GO_PHERENCE_DISABLE_NVIDIA=1 go run ./cmd/image/ideogram4gen \
		-model $(IDEOGRAM4_MODEL) \
		-prompt "$$(cat $(IDEOGRAM4_PROMPT_FILE))" \
		-out $(IDEOGRAM4_OUT) \
		-width $(IDEOGRAM4_WIDTH) \
		-height $(IDEOGRAM4_HEIGHT) \
		-steps $(IDEOGRAM4_STEPS) \
		-guidance $(IDEOGRAM4_GUIDANCE) \
		-mu $(IDEOGRAM4_MU) \
		-std $(IDEOGRAM4_STD) \
		-seed $(IDEOGRAM4_SEED) \
		-timing

ideogram4-cat-open:
	@echo $(IDEOGRAM4_OUT)

IDEOGRAM4_SWEEP_STEPS ?= 2
IDEOGRAM4_SWEEP_WINDOWS ?= 0 2 4 8
IDEOGRAM4_SWEEP_STARTS ?= 0
IDEOGRAM4_SWEEP_CSV ?= $(TMPDIR)/ideogram4/residency_sweep.csv
.PHONY: ideogram4-residency-sweep
ideogram4-residency-sweep:
	IDEOGRAM4_SWEEP_STEPS='$(IDEOGRAM4_SWEEP_STEPS)' \
	IDEOGRAM4_SWEEP_WINDOWS='$(IDEOGRAM4_SWEEP_WINDOWS)' \
	IDEOGRAM4_SWEEP_STARTS='$(IDEOGRAM4_SWEEP_STARTS)' \
	IDEOGRAM4_SWEEP_CSV='$(IDEOGRAM4_SWEEP_CSV)' \
	IDEOGRAM4_MODEL='$(IDEOGRAM4_MODEL)' \
	IDEOGRAM4_PROMPT_FILE='$(IDEOGRAM4_PROMPT_FILE)' \
	IDEOGRAM4_WIDTH='$(IDEOGRAM4_WIDTH)' \
	IDEOGRAM4_HEIGHT='$(IDEOGRAM4_HEIGHT)' \
	IDEOGRAM4_GUIDANCE='$(IDEOGRAM4_GUIDANCE)' \
	IDEOGRAM4_MU='$(IDEOGRAM4_MU)' \
	IDEOGRAM4_STD='$(IDEOGRAM4_STD)' \
	IDEOGRAM4_SEED='$(IDEOGRAM4_SEED)' \
	IDEOGRAM4_GPU_RESIDENCY='$(IDEOGRAM4_GPU_RESIDENCY)' \
	IDEOGRAM4_FULL_LAYER='$(IDEOGRAM4_FULL_LAYER)' \
	./scripts/ideogram4_residency_sweep.sh

# Resolution-aware Ideogram presets for the local RTX 3060 12GB profile.
# 256px can use aggressive asymmetric residency. 512px needs reduced residency
# because the larger activation/attention buffers exceed VRAM with 256px defaults.
IDEOGRAM4_512_STEPS ?= 4
IDEOGRAM4_VAE_PROBE_HEIGHT ?= 512
IDEOGRAM4_VAE_PROBE_WIDTH ?= 512
.PHONY: ideogram4-cat-gpu-256 ideogram4-cat-gpu-512 ideogram4-vae-probe

ideogram4-cat-gpu-256:
	$(MAKE) ideogram4-cat-gpu \
		IDEOGRAM4_WIDTH=256 \
		IDEOGRAM4_HEIGHT=256 \
		IDEOGRAM4_LAYER_CACHE_WINDOW=21 \
		IDEOGRAM4_LAYER_CACHE_WINDOW_COND=34 \
		IDEOGRAM4_LAYER_CACHE_WINDOW_UNCOND=9

ideogram4-cat-gpu-512:
	$(MAKE) ideogram4-cat-gpu \
		IDEOGRAM4_WIDTH=512 \
		IDEOGRAM4_HEIGHT=512 \
		IDEOGRAM4_STEPS=$(IDEOGRAM4_512_STEPS) \
		IDEOGRAM4_OUT=$(TMPDIR)/ideogram4/cat_comfy_prompt_512.png \
		IDEOGRAM4_LAYER_CACHE_WINDOW=0 \
		IDEOGRAM4_LAYER_CACHE_WINDOW_COND=16 \
		IDEOGRAM4_LAYER_CACHE_WINDOW_UNCOND=0

ideogram4-vae-probe:
	go run ./cmd/image/ideogram4vaeprobe \
		-model $(IDEOGRAM4_MODEL) \
		-width $(IDEOGRAM4_VAE_PROBE_WIDTH) \
		-height $(IDEOGRAM4_VAE_PROBE_HEIGHT) \
		-gpu -gpu-stats

DIFFUSIONGEMMA_REPO ?= google/diffusiongemma-26B-A4B-it
DIFFUSIONGEMMA_MODEL ?= models/diffusiongemma-26B-A4B-it

.PHONY: diffusiongemma-download-metadata diffusiongemma-download

diffusiongemma-download-metadata:
	python3 scripts/download_diffusiongemma.py --repo $(DIFFUSIONGEMMA_REPO) --out $(DIFFUSIONGEMMA_MODEL) --metadata-only

diffusiongemma-download:
	python3 scripts/download_diffusiongemma.py --repo $(DIFFUSIONGEMMA_REPO) --out $(DIFFUSIONGEMMA_MODEL)

DIFFUSIONGEMMA_PROMPT_IDS ?= 2
DIFFUSIONGEMMA_PROMPT ?= hi
DIFFUSIONGEMMA_MAX_NEW ?= 16
DIFFUSIONGEMMA_CANVAS ?= 0
DIFFUSIONGEMMA_SEED ?= 1
DIFFUSIONGEMMA_MOCK_TOKEN ?= 4

.PHONY: diffusiongemma-inspect diffusiongemma-inspect-json diffusiongemma-run-scaffold diffusiongemma-run-mock diffusiongemma-run-cpu

diffusiongemma-inspect:
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL)

diffusiongemma-inspect-json:
	go run ./cmd/diffusiongemmainspect -model $(DIFFUSIONGEMMA_MODEL) -json

diffusiongemma-run-scaffold:
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt-ids $(DIFFUSIONGEMMA_PROMPT_IDS) -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED)

diffusiongemma-run-mock:
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt '$(DIFFUSIONGEMMA_PROMPT)' -mock-token $(DIFFUSIONGEMMA_MOCK_TOKEN) -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -decode

diffusiongemma-run-cpu:
	go run ./cmd/diffusiongemmarun -model $(DIFFUSIONGEMMA_MODEL) -prompt-ids $(DIFFUSIONGEMMA_PROMPT_IDS) -max-new $(DIFFUSIONGEMMA_MAX_NEW) -canvas $(DIFFUSIONGEMMA_CANVAS) -seed $(DIFFUSIONGEMMA_SEED) -cpu-dispatcher

DIFFUSIONGEMMA_REF_OUT ?= $(TMPDIR)/diffusiongemma/reference.json
DIFFUSIONGEMMA_REF_PROMPT ?= Why is the sky blue?
DIFFUSIONGEMMA_REF_MAX_NEW ?= 64
DIFFUSIONGEMMA_REF_STEPS ?= 48

.PHONY: diffusiongemma-reference-dry-run diffusiongemma-reference

diffusiongemma-reference-dry-run:
	python3 scripts/diffusiongemma_reference.py --model $(DIFFUSIONGEMMA_MODEL) --dry-run --out $(DIFFUSIONGEMMA_REF_OUT)

diffusiongemma-reference:
	python3 scripts/diffusiongemma_reference.py --model $(DIFFUSIONGEMMA_MODEL) --prompt '$(DIFFUSIONGEMMA_REF_PROMPT)' --max-new-tokens $(DIFFUSIONGEMMA_REF_MAX_NEW) --max-denoising-steps $(DIFFUSIONGEMMA_REF_STEPS) --out $(DIFFUSIONGEMMA_REF_OUT)
