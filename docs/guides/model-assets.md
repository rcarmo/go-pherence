# Model assets

[Command index](commands.md) | [Runtime tuning](tuning.md)

## Model asset downloads

Downloaded weights, configs and tokenizers live under `checkpoints/`, which is entirely ignored by Git. All implementation source lives under `model/`. Use the helper script directly or via Make targets:

```bash
make models-list
make models-download-small
make models-download-qwen
make models-download-qwen3tts
make models-download-lfm2
make models-download-minicpmv
make models-download-minicpmo
make minicpmv-assets-check
make models-download-gemma4
make models-download-speaker
make models-download-one MODEL=qwen3.6-27b-mlx4-mtp
```

Set `CHECKPOINTS_DIR=/path/to/weights` on Make targets to change the asset root. Python helpers accept `--checkpoints-dir`. Existing `MODELS_DIR` and `--models-dir` overrides remain compatibility aliases; new examples use the checkpoint names.

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
  --checkpoint checkpoints/speechbrain-ecapa-voxceleb/embedding_model.ckpt \
  --output checkpoints/speaker-ecapa-voxceleb.safetensors \
  --dump-keys
```

For gated repositories, set `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`. If an upstream repo is renamed, override it without editing the script:

```bash
python3 scripts/download_models.py --only gemma4-e4b-it-4bit --repo gemma4-e4b-it-4bit=org/repo
```

## Older checkouts

BERT, Whisper, speaker (including Community-1) and OmniVoice source moved from
`models/<family>` to `model/<family>`. Update downstream Go imports accordingly;
there are no forwarding packages at the old paths.

After pulling the source move, rename the remaining repository-local `models/`
directory to `checkpoints/` if that destination does not already exist. If both
exist, merge assets deliberately rather than overwriting either directory. The
old root remains ignored by Git so residual weights cannot be committed by
accident. New downloads default to `checkpoints/`.

User-specified absolute paths and external stores such as `/workspace/models`,
`/opt/models` and a separate `llama.cpp/models` tree are unchanged. Environment
variables continue to accept arbitrary paths. Historical benchmark records
retain their original paths and source revision; translate those paths when
replaying an older command against the new layout.

## Keeping the layout consistent

`make model-layout-check` checks worktree source, new templates, executable
scaffolding, live fixtures and fenced documentation examples. It also tests
Make/Python download defaults and compatibility aliases without downloading
anything. `make docs-check` includes it, `make host-check` runs it first, and
ordinary `go test ./...` runs the Go-only source/path guard.

The guard scans foreign-platform files as text rather than executing them.
Checkpoint payloads, external stores and dated benchmark records stay outside
that scan. New benchmark scripts are checked; the few frozen reproducer scripts
have explicit, documented exceptions. The GitHub layout workflow runs these
checks without model assets on pushes and pull requests.
