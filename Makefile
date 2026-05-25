TMPDIR ?= /workspace/tmp
GOTMPDIR ?= /workspace/tmp
PYTHON ?= python3
MODELS_DIR ?= models
MODEL ?=
MODEL_DOWNLOAD_FLAGS ?=
export TMPDIR GOTMPDIR

.PHONY: all build test test-cpu clean server chat gen vet models-list models-download models-download-small models-download-qwen models-download-gemma4 models-download-one hunyuan3d-inventory hunyuan3d-image-fixture hunyuan3d-conditioner-fixture hunyuan3d-denoiser-fixture hunyuan3d-lowstep-fixture

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

models-download-gemma4:
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) --group gemma4 $(MODEL_DOWNLOAD_FLAGS)

models-download-one:
	@if [ -z "$(MODEL)" ]; then echo "usage: make models-download-one MODEL=qwen3.6-27b-mlx4-mtp"; exit 2; fi
	$(PYTHON) scripts/download_models.py --models-dir $(MODELS_DIR) --only $(MODEL) $(MODEL_DOWNLOAD_FLAGS)

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
HUNYUAN3D_CONDITIONER_FIXTURE ?= /workspace/tmp/hunyuan3d-conditioner-fixture.json
HUNYUAN3D_CONDITIONER_FLAGS ?=
HUNYUAN3D_DENOISER_FIXTURE ?= /workspace/tmp/hunyuan3d-denoiser-step-fixture.json
HUNYUAN3D_DENOISER_FLAGS ?=
HUNYUAN3D_LOWSTEP_FIXTURE ?= /workspace/tmp/hunyuan3d-lowstep-latents-fixture.json
HUNYUAN3D_LOWSTEP_FLAGS ?=

hunyuan3d-inventory:
	$(PYTHON) scripts/hunyuan3d_fixture_inventory.py --repo $(HUNYUAN3D_REPO) --subfolder $(HUNYUAN3D_SUBFOLDER) --out $(HUNYUAN3D_INVENTORY) $(HUNYUAN3D_INVENTORY_FLAGS)

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
