# Model assets

[Command index](commands.md) | [Runtime tuning](tuning.md)

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
make minicpmv-assets-check
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
