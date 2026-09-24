# NVIDIA Nemotron voice-synthesis roadmap

NVIDIA's published NemotronLabs VoiceChat 11B supports speech generation within a full-duplex conversational model. This roadmap tracks that **speech-output capability**; a separately packaged Nemotron text-to-speech checkpoint has not been identified. No weights have been downloaded or run, and go-pherence has no VoiceChat inference implementation.

## Pinned candidate and boundary

- [NVIDIA-NemotronLabs-VoiceChat-11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B/tree/443794ea956ef0065f001967ffd00e77f519cb39), revision `443794ea956ef0065f001967ffd00e77f519cb39`. The model card describes streaming speech understanding and generation, full-duplex turn-taking and tool calling; its reported turn-taking latency is an upstream claim, not a go-pherence measurement.
- Hub metadata identifies `openmdw-1.1` as the licence and lists an 11B-class model. Review licence, deployment and redistribution terms before checkpoint use. The card describes a Nemotron Nano V2 backbone and a speech decoder/codec; it does not establish a standalone text-to-speech API for our integration.
- [Nemotron 3 Diarization](nemotron-3-diarization-roadmap.md) is an independent speaker-attribution model. NVIDIA's `nemotron-speech-streaming-en-0.6b` is an ASR model, not a TTS model. Neither serves as a voice-synthesis oracle.

## Gates

1. Resolve the exact source, processor, audio codec, dialogue/turn-taking and speaker-identity contracts. Confirm whether a public standalone TTS path exists before specifying a voice-only API; inspect manifests without reading weight payloads.
2. Obtain explicit approval for checkpoint and oracle execution. Capture independent bounded input/output fixtures covering audio input, streamed text/audio emission, interruption/barge-in, speaker continuity and tool-call turn transitions. Record model/source revision and artifact hashes.
3. Plan admission for 11B weights, codec and stream state on each target machine before implementation. Build an owned Go scalar path, then profile and qualify any checked SIMD/NVIDIA acceleration separately; do not substitute a Python/vLLM process for native inference.
4. Evaluate intelligibility, speaker/voice stability, latency and cancellation on approved held-out conversations. Keep speech output opt-in and separate from the [Chatterbox](chatterbox-tts-roadmap.md), [Pocket TTS](../models/pocket-tts-support.md) and [Qwen3-TTS](../models/qwen3-tts-support.md) paths.

This is model inspection and implementation planning only. It does not imply native streaming voice generation, standalone TTS, trained-quality parity or a supported service.
