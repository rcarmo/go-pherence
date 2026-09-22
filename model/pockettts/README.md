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
- direct `TrainableTTS.forward` parity for combined EOS, normalized diagonal and minimal-stop-gradient `s→t` losses, every parameter/input gradient and shared `w_s_t` weighting network;
- full-model AdamW, EMA and directory-durable checkpoint/resume state, including mutable latent statistics and fixed timestep-frequency buffers;
- strict upstream frozen-Mimi latent-cache admission: exact metadata/hash and shard naming, one finite F32 `[frames,channels]` tensor per row, bounded immutable content digests, deterministic atomic F32 shard output, and fixed-overlap fresh-prefix stitching.

The deterministic one-step correctness graph is complete for one-row F32 training with dropout disabled and explicit sampled inputs. The initial allocation-first pass adds request-owned `TrainingWorkspace`/`PocketTrainingStepInto`, zero-allocation warm AdamW/EMA, and regression ceilings; the tiny full step uses 68.5% fewer allocated bytes and 45.9% fewer allocations. Upstream-generated frozen Mimi caches are now directly admitted and native F32 shards round-trip through upstream `safetensors`; raw-audio Mimi encoding remains deferred. Production-size cache shapes are next before further arena/SIMD work, followed by teacher/student distillation. See [`docs/models/pocket-tts-support.md`](../../docs/models/pocket-tts-support.md) for provenance, limits and the inference/training sequence.
