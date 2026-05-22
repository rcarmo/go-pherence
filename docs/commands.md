# Commands

## Model asset downloads

Downloaded model assets live under `models/`, which is ignored by git except for source packages. Use the helper script directly or via Make targets:

```bash
make models-list
make models-download-small
make models-download-qwen
make models-download-gemma4
make models-download-one MODEL=qwen3.6-27b-mlx4-mtp
```

Forward extra options through `MODEL_DOWNLOAD_FLAGS`:

```bash
make models-download-one MODEL=qwen3.6-27b-mlx4-mtp MODEL_DOWNLOAD_FLAGS='--force'
make models-list MODEL_DOWNLOAD_FLAGS='--group qwen'
python3 scripts/download_models.py --dry-run --group gemma4
```

The downloader uses `huggingface_hub.snapshot_download`; install it with:

```bash
python3 -m pip install huggingface_hub
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
