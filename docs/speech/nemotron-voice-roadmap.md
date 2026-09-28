# NVIDIA Nemotron voice-synthesis roadmap

NVIDIA's published NemotronLabs VoiceChat 11B supports speech generation within a full-duplex conversational model. This roadmap tracks that **speech-output capability**; a separately packaged Nemotron text-to-speech checkpoint has not been identified. No weights have been downloaded or run, and go-pherence has no VoiceChat inference implementation.

## Pinned candidate and boundary

- [NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/tree/443794ea956ef0065f001967ffd00e77f519cb39), revision `443794ea956ef0065f001967ffd00e77f519cb39`. The model card describes streaming speech understanding and generation, full-duplex turn-taking and tool calling; its reported turn-taking latency is an upstream claim, not a go-pherence measurement.
- Hub metadata at the pinned revision identifies `openmdw-1.1` as the licence and lists an 11B-class model. Its licence grants use of the model materials subject to its terms; redistribution requires retaining the licence and applicable copyright/origin notices. It imposes no obligations on generated outputs, but leaves third-party rights/permissions to the user. Review deployment terms before checkpoint use. The card describes a Nemotron Nano V2 backbone and a speech decoder/codec; it does not establish a standalone text-to-speech API for our integration.
- [Nemotron 3 Diarization](nemotron-3-diarization-roadmap.md) is an independent speaker-attribution model. NVIDIA's `nemotron-speech-streaming-en-0.6b` is an ASR model, not a TTS model. Neither serves as a voice-synthesis oracle.

## Pinned metadata inspection

The public Hub file list at revision `443794ea956ef0065f001967ffd00e77f519cb39` contains `config.json`, `README.md`, `LICENSE`, RNNT tokenizer files and one `model.safetensors`. Only those text/config metadata files were inspected; no weight payload was read. The card describes 16 kHz user audio input and 22.05 kHz agent audio output. `config.json` specifies 80 ms frames, a speech-generation codec with 31 quantizers, 1,024-entry codebooks and `wav_to_token_ratio: 1764`, and a three-second audio prompt setting. These are checkpoint/config values, not a verified Go decoder contract.

The upstream offline example on the `NVIDIA-NeMo/Speech` `nemotron-labs-voicechat` branch requires `--checkpoint`, `--wav` and `--output-dir`, loads mono input at 16 kHz, calls `model.offline_inference()` and saves agent text/audio. Its optional system prompt is dialogue context, not a documented text-only speech-generation input. The card's interactive path uses NVIDIA's CUDA/Triton/vLLM WebSocket container. The metadata and example provide no public standalone TTS invocation. Treat the card's mention of TTS as a capability within full-duplex VoiceChat, not a separate text-only API.

## Gates

1. Resolve the exact source, processor, audio codec, dialogue/turn-taking and speaker-identity contracts, including codec token timing and the speech decoder's text/audio alignment. Confirm whether a public standalone TTS path exists before specifying a voice-only API; inspect manifests without reading weight payloads.
2. Obtain explicit approval for checkpoint and oracle execution. Capture independent bounded input/output fixtures covering audio input, streamed text/audio emission, interruption/barge-in, speaker continuity and tool-call turn transitions. Record model/source revision and artifact hashes.
3. Plan admission for 11B weights, codec and stream state on each target machine before implementation. Build an owned Go scalar path, then profile and qualify any checked SIMD/NVIDIA acceleration separately; do not substitute a Python/vLLM process for native inference.
4. Evaluate intelligibility, speaker/voice stability, latency and cancellation on approved held-out conversations. Keep speech output opt-in and separate from the [Chatterbox](chatterbox-tts-roadmap.md), [Pocket TTS](../models/pocket-tts-support.md) and [Qwen3-TTS](../models/qwen3-tts-support.md) paths.

This is model inspection and implementation planning only. It does not imply native streaming voice generation, standalone TTS, trained-quality parity or a supported service.
