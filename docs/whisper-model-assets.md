# Whisper model assets in use

This project currently uses local Hugging Face-style Whisper safetensors assets under `models/` for `cmd/diarize-vtt` and related Whisper tests/tools.

## Production/default translated VTT model

| Field | Value |
|---|---|
| Upstream model name | `openai/whisper-large-v3` |
| Local directory | `models/whisper-large-v3-hf/` |
| Weight file | `models/whisper-large-v3-hf/model.safetensors` |
| Local weight size | ~2.9 GiB |
| Tokenizer | `models/whisper-large-v3-hf/tokenizer.json` |
| CLI size flag | `-size large-v3` |
| Go config | `models/whisper.LargeV3()` |
| Status | **Default/recommended** for translated WebVTT output |

Shape/config used by Go:

| Parameter | Value |
|---|---:|
| Mel bins | 128 |
| Max encoder length | 3000 |
| Encoder layers | 32 |
| Encoder d_model | 1280 |
| Encoder heads | 20 |
| Encoder FFN dim | 5120 |
| Decoder layers | 32 |
| Decoder d_model | 1280 |
| Decoder heads | 20 |
| Decoder FFN dim | 5120 |
| Vocab size | 51866 |
| Max decoder length | 448 |
| Head dim | 64 |

Default `diarize-vtt` usage:

```bash
go run ./cmd/diarize-vtt \
  -input meeting.m4a \
  -output meeting.vtt \
  -model models/whisper-large-v3-hf/model.safetensors \
  -size large-v3 \
  -task translate \
  -language pt
```

Notes:

- This is the default `cmd/diarize-vtt` model path.
- The workflow loads `tokenizer.json` from the same directory as `model.safetensors`.
- This is currently kept as the production/default path because translation behavior is known-good compared with the turbo candidate.

## Faster research candidate

| Field | Value |
|---|---|
| Upstream model name | `openai/whisper-large-v3-turbo` |
| Local directory | `models/whisper-large-v3-turbo-hf/` |
| Weight file | `models/whisper-large-v3-turbo-hf/model.safetensors` |
| Local weight size | ~1.6 GiB |
| Config | `models/whisper-large-v3-turbo-hf/config.json` |
| Generation config | `models/whisper-large-v3-turbo-hf/generation_config.json` |
| Preprocessor config | `models/whisper-large-v3-turbo-hf/preprocessor_config.json` |
| Tokenizer | `models/whisper-large-v3-turbo-hf/tokenizer.json` |
| CLI size flag | `-size turbo` or `-size large-v3-turbo` |
| Go config | `models/whisper.LargeV3Turbo()` |
| Status | **Research/validation only**, not the default |

Shape/config from local `config.json` and Go:

| Parameter | Value |
|---|---:|
| Model type | `whisper` |
| Mel bins | 128 |
| Max source positions | 1500 |
| Encoder layers | 32 |
| Encoder d_model | 1280 |
| Encoder heads | 20 |
| Decoder layers | 4 |
| Decoder d_model | 1280 |
| Decoder heads | 20 |
| Vocab size | 51866 |
| Max target positions | 448 |

Important token IDs from `generation_config.json`:

| Token/config | Value |
|---|---:|
| `decoder_start_token_id` | 50258 |
| `translate` | 50359 |
| `transcribe` | 50360 |
| `no_timestamps_token_id` | 50364 |
| `forced_decoder_ids` | `[[1, null], [2, 50360]]` |

Test command:

```bash
go run ./cmd/diarize-vtt \
  -input meeting.m4a \
  -output meeting.vtt \
  -model models/whisper-large-v3-turbo-hf/model.safetensors \
  -size turbo \
  -task translate \
  -language en
```

Current validation status:

- Loads successfully and is shape-valid.
- Stress RTF observed around `0.54–0.56`, materially faster than full large-v3.
- Byte-level tokenizer decoding has been fixed.
- `generation_config` suppression tokens are wired.
- Translation parity was checked on a Portuguese voice-memo clip (`/workspace/tmp/voice_memo_300_12.wav`). Transformers and Go both produce English with `task=translate` when prompted with `language=en`; `language=pt`/`portuguese` degenerates or transcribes Portuguese. Turbo is therefore not a drop-in replacement for the current source-language CLI default; keep full `large-v3` as default until this language-token behavior is made ergonomic and validated on more clips.

## Related helper scripts

| Script | Purpose |
|---|---|
| `scripts/whisper_transformers_reference.py` | Optional Hugging Face Transformers reference decode for prompt/task parity checks, especially turbo translation behavior. |
| `scripts/download_models.py` | General model asset downloader; current Whisper assets were downloaded/staged manually/local workflow rather than listed as a first-class download group. |

## Summary for handoff

Use these names/paths when referring to current Whisper weights:

```text
Default:  openai/whisper-large-v3
Path:     models/whisper-large-v3-hf/model.safetensors
Tokenizer models/whisper-large-v3-hf/tokenizer.json
Go size:  large-v3

Research: openai/whisper-large-v3-turbo
Path:     models/whisper-large-v3-turbo-hf/model.safetensors
Tokenizer models/whisper-large-v3-turbo-hf/tokenizer.json
Go size:  turbo / large-v3-turbo
```
