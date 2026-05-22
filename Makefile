TMPDIR ?= /workspace/tmp
GOTMPDIR ?= /workspace/tmp
PYTHON ?= python3
MODELS_DIR ?= models
MODEL ?=
MODEL_DOWNLOAD_FLAGS ?=
export TMPDIR GOTMPDIR

.PHONY: all build test test-cpu clean server chat gen vet models-list models-download models-download-small models-download-qwen models-download-gemma4 models-download-one

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
