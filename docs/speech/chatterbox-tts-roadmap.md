# Chatterbox TTS roadmap

Resemble AI's Chatterbox is a candidate for a separate native voice-synthesis backend. No Chatterbox checkpoint has been downloaded or run for this roadmap. It is not implemented in go-pherence.

## Pinned candidates

- [Chatterbox](https://huggingface.co/ResembleAI/chatterbox/tree/5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18): published as English zero-shot TTS with reference-audio conditioning and generation controls. The card describes a 500M-parameter model.
- [Chatterbox-Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo/tree/749d1c1a46eb10492095d68fbcf55691ccf137cd): published as a 350M-parameter English TTS model with a one-step speech-token-to-mel decoder and reference-audio example. Turbo is a separate candidate, not an interchangeable checkpoint.
- Both Hub repositories report `mit` licence metadata. Inspect the applicable source, checkpoint and audio-use terms before importing artifacts or publishing generated examples. Multilingual Chatterbox variants require separate fixtures and approval.

## Gates

1. Select one exact model and source-code revision. Inspect processor, tokenizer, speaker/reference-audio frontend, generated token/mel and waveform decoder contracts; record checkpoint inventory and hashes without fetching weights.
2. Obtain approval before downloading or executing checkpoint weights or reference voice clips. Freeze independent oracle fixtures for text preparation, reference-audio conditioning, first tokens, acoustic latents and first waveform frames. Use consented or synthetic reference audio; keep evaluation audio and training data rights explicit.
3. Implement a scalar Go inference path with bounded context, explicit voice conditioning, memory admission and cancelled-run ownership. Qualify operator parity and deterministic output before accelerating measured bottlenecks using checked SIMD.
4. Compare output quality, speaker similarity and latency against the pinned oracle on approved, held-out speech. Keep the existing [Pocket TTS](../models/pocket-tts-support.md) and [Qwen3-TTS](../models/qwen3-tts-support.md) work separate; do not transfer their parity or throughput evidence.

There is no native Chatterbox inference, released-model parity, voice-cloning quality or production throughput evidence here.
