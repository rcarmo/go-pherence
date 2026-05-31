TMPDIR ?= /workspace/tmp
GOTMPDIR ?= /workspace/tmp
PYTHON ?= python3
MODELS_DIR ?= models
MODEL ?=
MODEL_DOWNLOAD_FLAGS ?=
export TMPDIR GOTMPDIR

.PHONY: all build test test-cpu test-model-coverage model-coverage model-coverage-json model-coverage-markdown model-coverage-csv model-coverage-snapshot model-coverage-runtime-roadmap model-coverage-runtime-roadmap-json model-coverage-pending model-coverage-references-pending model-coverage-runtime-pending model-coverage-parity-pending model-coverage-readiness-pending model-coverage-references-gate model-coverage-runtime-gate model-coverage-parity-gate model-coverage-readiness-gate clean server chat gen vet models-list models-download models-download-small models-download-qwen models-download-qwen3tts models-download-lfm2 models-download-gemma4 models-download-speaker models-download-one qwen3tts-inspect qwen3tts-fixture-coverage lfm2-inspect lfm2-fixture-coverage hunyuan3d-fixture-env hunyuan3d-inventory hunyuan3d-inspect hunyuan3d-image-fixture hunyuan3d-conditioner-fixture hunyuan3d-denoiser-fixture hunyuan3d-lowstep-fixture hunyuan3d-mesh-fixture trellis2-fixture-env trellis2-inventory trellis2-lowstep-fixture trellis2-ovoxel-inspect

all: build

build: gen server chat

gen:
	go build -o bin/llmgen ./cmd/llmgen

server:
	go build -o bin/llmserver ./cmd/llmserver

chat:
	go build -o bin/llmchat ./cmd/llmchat

test:
	go test -count=1 -timeout=120s ./loader/... ./model/... ./models/bert/... ./backends/nvidia/... ./backends/placement/... ./backends/simd/... ./backends/vulkan/... ./runtime/... ./tensor/...

test-cpu:
	GO_PHERENCE_DISABLE_NVIDIA=1 GO_PHERENCE_VULKAN_ALLOW_CPU=0 go test -count=1 -timeout=120s ./loader/... ./model/... ./models/bert/... ./backends/nvidia/... ./backends/placement/... ./backends/simd/... ./backends/vulkan/... ./runtime/... ./tensor/...

test-model-coverage:
	go test -count=1 -timeout=120s ./docs ./loader/safetensors ./model/qwen3tts ./model/lfm2 ./cmd/qwen3ttsinspect ./cmd/lfm2inspect ./cmd/modelcoverage
	go vet ./docs ./loader/safetensors ./model/qwen3tts ./model/lfm2 ./cmd/qwen3ttsinspect ./cmd/lfm2inspect ./cmd/modelcoverage
	go run ./cmd/modelcoverage -references-only -fail-pending
	go run ./cmd/modelcoverage -parity-only -fail-pending
	go run ./cmd/modelcoverage -readiness-only -fail-pending
	go run ./cmd/modelcoverage -min-percent $(MODEL_COVERAGE_MIN_PERCENT)

MODEL_COVERAGE_FAMILY ?=
MODEL_COVERAGE_MIN_PERCENT ?= 90

model-coverage:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),)

model-coverage-json:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -json

model-coverage-markdown:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -markdown

model-coverage-csv:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -csv

model-coverage-snapshot:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -snapshot

model-coverage-runtime-roadmap:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-roadmap

model-coverage-runtime-roadmap-json:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-roadmap-json

model-coverage-pending:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -pending-only

model-coverage-references-pending:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -references-only -pending-only

model-coverage-runtime-pending:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-only -pending-only

model-coverage-parity-pending:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -parity-only -pending-only

model-coverage-readiness-pending:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -readiness-only -pending-only

model-coverage-references-gate:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -references-only -fail-pending

model-coverage-runtime-gate:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -runtime-only -fail-pending

model-coverage-parity-gate:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -parity-only -fail-pending

model-coverage-readiness-gate:
	go run ./cmd/modelcoverage $(if $(MODEL_COVERAGE_FAMILY),-family $(MODEL_COVERAGE_FAMILY),) -readiness-only -fail-pending

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
	go run ./cmd/qwen3ttsinspect -model $(QWEN3TTS_MODEL) -text "$(QWEN3TTS_TEXT)" -speaker $(QWEN3TTS_SPEAKER) -language $(QWEN3TTS_LANGUAGE) $(QWEN3TTS_INSPECT_FLAGS)

qwen3tts-fixture-coverage:
	@if [ -z "$(QWEN3TTS_MODEL)" ]; then echo "usage: make qwen3tts-fixture-coverage QWEN3TTS_MODEL=models/qwen3-tts-0.6b-customvoice [QWEN3TTS_FIXTURE=model/qwen3tts/testdata/customvoice_prompt_fixture.json]"; exit 2; fi
	go run ./cmd/qwen3ttsinspect -model $(QWEN3TTS_MODEL) -fixture $(QWEN3TTS_FIXTURE) $(QWEN3TTS_FIXTURE_FLAGS)

lfm2-inspect:
	@if [ -z "$(LFM2_MODEL)" ]; then echo "usage: make lfm2-inspect LFM2_MODEL=models/lfm2.5-8b-a1b"; exit 2; fi
	go run ./cmd/lfm2inspect -model $(LFM2_MODEL) $(LFM2_INSPECT_FLAGS)

lfm2-fixture-coverage:
	@if [ -z "$(LFM2_MODEL)" ]; then echo "usage: make lfm2-fixture-coverage LFM2_MODEL=models/lfm2.5-8b-a1b [LFM2_FIXTURE=model/lfm2/testdata/lfm25_8b_a1b_metadata.json]"; exit 2; fi
	go run ./cmd/lfm2inspect -model $(LFM2_MODEL) -fixture $(LFM2_FIXTURE) $(LFM2_FIXTURE_FLAGS)

HUNYUAN3D_REPO ?= tencent/Hunyuan3D-2mini
HUNYUAN3D_SUBFOLDER ?= hunyuan3d-dit-v2-mini
HUNYUAN3D_INVENTORY ?= /workspace/tmp/hunyuan3d-mini-inventory.json
HUNYUAN3D_INVENTORY_FLAGS ?= --include-tensors
HUNYUAN3D_IMAGE_FIXTURE ?= /workspace/tmp/hunyuan3d-image-preprocess-fixture.json
HUNYUAN3D_IMAGE ?=
HUNYUAN3D_IMAGE_FLAGS ?=
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
	go run ./cmd/hy3dinspect -config $(HUNYUAN3D_CONFIG) $(if $(HUNYUAN3D_CHECKPOINT),-safetensors $(HUNYUAN3D_CHECKPOINT),) $(HUNYUAN3D_INSPECT_FLAGS)

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
	go build -o /dev/null ./cmd/llmgen
	go build -o /dev/null ./cmd/llmserver
	go build -o /dev/null ./cmd/llmchat
	@echo "=== vet ==="
	go vet ./...
	@echo "=== unit tests ==="
	go test -count=1 -timeout=60s ./loader/... ./backends/nvidia/... ./backends/placement/... ./backends/simd/... ./backends/vulkan/... ./runtime/... ./tensor/...
	@echo "=== ok ==="
