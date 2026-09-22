# Pocket TTS

Native Go/SIMD implementation of Kyutai Pocket TTS, pinned to upstream commit `0acce6b2f390150267557770d2098c5caa9a18ac`.

Implemented:

- released YAML topology parsing;
- 12.5 Hz, 24 kHz mono and 1,920-sample/frame geometry;
- BF16/F32 checkpoint inventory and shape validation;
- Unigram/Metaspace tokenizer with UTF-8 byte fallback;
- SIMD affine and LSD updates;
- SIMD six-layer FlowLM stateless/streaming execution with released hidden/EOS parity;
- native preset voice-state K/V import and released first-latent parity;
- SIMD `SimpleMLPAdaLN` flow-head execution with released-weight parity;
- stateful SIMD Mimi decode with released 1,920-sample waveform parity;
- autoregressive generation contracts and exclusive PCM16 mono WAV output;
- aligned-manifest admission, exact EOS/FlowMatching/LSD-diagonal losses and analytic gradients for a frozen-backbone affine training topology;
- F32-owned `SimpleMLPAdaLN` reverse-mode gradients, exact forward-mode time JVPs and reverse-over-JVP mixed derivatives, with pinned upstream PyTorch parity;
- F32-owned stateless causal-transformer backward through bounded attention, adjacent-pair RoPE, layer scales, tanh-GELU FFN and final LayerNorm;
- exact one-row training layout and gradients for BOS-before-voice, voice projection, text embeddings, shifted audio projection, gathered audio conditions and EOS head;
- deterministic AdamW, EMA and directory-durable checkpoint/resume state for the frozen topology.

The normalized LSD `s→t` term now includes exact mixed derivatives and the upstream minimal stop-gradient endpoint rule. Transformer and conditioning backward are implemented. Combined EOS + diagonal + `s→t` loss accumulation and complete one-step parity are still open, followed by latent precomputation and teacher/student distillation. See [`docs/models/pocket-tts-support.md`](../../docs/models/pocket-tts-support.md) for provenance, limits and the inference/training sequence.
