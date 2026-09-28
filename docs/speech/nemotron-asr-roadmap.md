# NVIDIA Nemotron ASR roadmap

NVIDIA's Nemotron streaming ASR checkpoints are candidates for a separate native transcription provider. Go-pherence has no Nemotron ASR implementation. No checkpoint weights were downloaded or executed for this roadmap.

## Pinned candidates

- [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/ea30d66debe3740a08b573244286791d423d6b3e), revision `ea30d66debe3740a08b573244286791d423d6b3e`. The pinned [model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/ea30d66debe3740a08b573244286791d423d6b3e/README.md) describes a multilingual, cache-aware FastConformer/RNN-T with language-ID prompting, configurable streaming chunks, punctuation and capitalisation. It claims 40 language-locales, of which 32 are transcription-ready or broad-coverage. The inspected `config.json` names `Nemotron3_5AsrForRNNT`, 24 encoder layers and 128 mel bins. The model card SHA-256 is `a3344caadf796c084c6b90a9fa5978068fd45e3a019790bebe50489bb3c0f7b7`; config SHA-256 is `62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845`. Hub licence metadata links to OpenMDW-1.1.
- [Nemotron Speech Streaming EN 0.6B](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/tree/ebe59e5a817142986528bbbee5dba8db7b38ed50), revision `ebe59e5a817142986528bbbee5dba8db7b38ed50`. The English-only model card describes a cache-aware FastConformer/RNN-T, and its config names `NemotronAsrStreamingForRNNT`. Its card SHA-256 is `7701f4b2f1c16542c8ceb2b3a61dd144032898c17f4dc9cc1bbecda6972edec8`; config SHA-256 is `dffe850bc79ad2b0f8117804502b24d2c4a445aafbed4c1e40f8d78e0cb44065`. Hub licence metadata names the NVIDIA Open Model License. Review each model's actual licence terms and artifact rights before use.

These are distinct checkpoints and licences. Their cards use Parakeet-related architecture tags, but the [Parakeet TDT v3 roadmap](parakeet-asr-roadmap.md) targets a separate TDT decoder. [Nemotron 3 Diarization](nemotron-3-diarization-roadmap.md) supplies speaker attribution, while [NemotronLabs VoiceChat](nemotron-voice-roadmap.md) includes speech output; neither is an ASR oracle for these checkpoints.

## Gates

1. Choose one checkpoint and source revision for an implementation slice. Inspect processor, tokenizer, config, model-header inventory, frontend and chunk/cache contract without fetching tensor payloads. Freeze exact language/locale IDs, prompt and timestamp conventions and size limits.
2. Obtain explicit approval before downloading or running weights. Generate independent NeMo/Transformers fixtures for frontend, subsampling, cached attention/convolution state, RNN-T predictor/joiner and decoding. Test cold and warm chunks, different lookaheads, silence, language selection, cancellation and restart. Pin input audio hashes, token/text outputs, units and numerical tolerances.
3. Implement a scalar Go reference with owned bounded stream state, offline and incremental paths, deterministic overlap/timestamp handling and memory admission. Qualify parity before moving measured kernels to checked SIMD or NVIDIA backends.
4. Add explicit ASR provider selection to the [speech workflow](speech-integration.md). Compare accuracy on held-out labelled speech, language-specific word error rate, latency, throughput and recovery after cancellation. Preserve the existing Whisper/MOSS and optional speaker-provider contracts.

Published accuracy and latency claims belong to NVIDIA's evaluations. No native model execution, released-checkpoint parity, transcription quality or production readiness has been established here.
