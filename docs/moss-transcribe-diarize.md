# Native MOSS-Transcribe-Diarize support

## Pinned references

- Source: `OpenMOSS/MOSS-Transcribe-Diarize` at `b5ad0f8386b155ddb89f9332ba3ca71891900357`
- Checkpoint: `OpenMOSS-Team/MOSS-Transcribe-Diarize` at `0b0295acf3e6282a1692e1f6226faa32a453f7a2`
- Source and model card license: Apache-2.0
- Checkpoint payload: one BF16 safetensors shard, 683 tensors, 1,817,026,560 bytes

These revisions are the parity oracle. New upstream configurations must fail explicitly until separately audited.

## Native graph contract

```text
16 kHz mono samples
  -> Whisper log-mel frontend (80 bins, 400-sample FFT, 160-sample hop)
  -> independent padded 30 s chunks (480,000 samples / 3,000 mel frames)
  -> Whisper encoder (24 layers, width 1024, 16 heads, FFN 4096)
  -> retain 4 * audio_token_length encoder rows from each chunk
  -> concatenate chunks belonging to one recording
  -> merge each 4 adjacent rows: [T,1024] -> [T/4,4096]
  -> adaptor Linear(4096,1024)+SiLU+Linear(1024,1024)+LayerNorm(eps=1e-6)
  -> replace token 151671 audio placeholders in Qwen embeddings
  -> Qwen3 decoder (28 layers, width 1024, 16 Q heads, 8 KV heads,
     head dim 128, FFN 3072, RoPE theta 1,000,000, context 131,072)
  -> tied 151,936-row LM head, greedy decoding
  -> streaming [start][Sxx]text[end] parser
```

Processor constants are 12.5 audio tokens/s, four-frame merge, and numeric time markers every five seconds. Generation defaults to EOS `151645`, PAD/BOS `151643`, and 5,120 new tokens.

## SIMD-first policy

The first native target is amd64 AVX2/FMA. Every assembly-dispatched operation must have:

1. a portable scalar implementation,
2. deterministic randomized and edge-case scalar-oracle tests,
3. malformed-length/tail coverage,
4. a benchmark proving dispatch is useful,
5. a Transformers boundary fixture proving the approximation does not alter the model contract.

Existing AVX2/FMA SGEMM, vector, LayerNorm, FFT, convolution, and pooling paths are reused rather than duplicated. New assembly is added only for an uncovered measured hot operation. ARM64 NEON follows once the amd64 graph is green.

## Native CLI

Build and inspect CPU/SIMD and runtime-loaded NVIDIA PTX availability:

```bash
make moss-transcribe
bin/moss-transcribe -capabilities
```

The CLI selects validated NVIDIA stages automatically and warns before falling back to CPU/SIMD. Pass `-cpu` to force the scalar/SIMD oracle for parity checks and reproducible benchmarks. The first GPU slice keeps Whisper encoder projection weights resident; adaptor and Qwen3 generation remain on CPU until their independent parity gates pass.

Transcribe a PCM WAV file and emit parsed subtitles:

```bash
bin/moss-transcribe \
  -model-dir /path/to/MOSS-Transcribe-Diarize \
  -audio meeting.wav \
  -format srt \
  -output meeting.srt
```

Formats are `text`, `raw`, `json`, `srt`, and `ass`. The default instruction is the pinned upstream Chinese transcription/diarization prompt; `-prompt` supplies hotwords or an alternate instruction. Decoding is greedy only. The CLI rejects sampling controls, non-WAV containers, context overflow, and more than 5,120 new tokens explicitly.

## Verification and observed CPU performance

Run the opt-in real-checkpoint gates with:

```bash
make moss-transcribe-parity MOSS_TRANSCRIBE_MODEL_DIR=/path/to/MOSS-Transcribe-Diarize
```

On an Intel Core i7-12700 with `GOMAXPROCS=2`:

- AVX2 affine LayerNorm is approximately 3.5x faster than its scalar oracle (`~0.46 us` versus `~1.6 us` for width 1024).
- The AVX2/FMA 400-element float64 DFT dot is approximately 5.1x faster than scalar (`~31 ns` versus `~160 ns`).
- The full 24-layer audio boundary fixture completes in approximately 31 seconds.
- A one-second WAV through model load, audio encoder/adaptor, canonical 38-token multimodal prefill, and one generated token completes in approximately 37 seconds; model loading is approximately 4.6 seconds.
- The 11-second JFK real-speech parity gate completes in approximately 49 seconds with early EOS at 70 generated tokens.
- The initial hybrid RTX 3060 audio-boundary smoke (model load + JFK encode, zero decode tokens) was approximately 41.9 seconds versus 42.2 seconds forced CPU. Sharing each layer's activation upload across Q/K/V projections reduces it to approximately 41.1 seconds with identical output. This is not yet a useful end-to-end speedup because other per-layer activation transfers dominate; the ≥2x gate remains pending full encoder/adaptor/Qwen residency.
- CPU profiling attributes roughly 75% of sampled CPU time to the existing AVX2/FMA `SgemmNT`/`SgemmNN` kernels; exact GELU is the next non-GEMM hotspot at roughly 8% cumulative CPU.

Pinned Transformers 4.57.1 parity observations:

- log-mel frontend: bit-identical fixture;
- Whisper widened-BF16 boundary maximum: `8.03e-5`;
- temporal merge/adaptor widened maximum: `1.10e-5`;
- canonical Qwen3 selected/top-logit widened maximum: `1.34e-5`, with matching argmax;
- JFK real speech: all 70 generated IDs, raw transcript, three `S01` segments, and six timestamps match Transformers exactly;
- actual upstream BF16 differences are separately bounded because the native CPU graph widens BF16 checkpoint weights and computes activations in F32.

Speaker labels such as `S01` remain recording-local generative labels, not cross-recording identities.

## Readiness

The native CLI executes the complete checkpoint graph and has real-checkpoint frontend, encoder, adaptor, insertion, selected-logit, and exact real-speech transcript/speaker/timestamp gates. The pinned CPU implementation is ready for greedy PCM-WAV inference. Sampling, video/container decoding, and stable cross-recording speaker identity remain explicitly unsupported; longer recordings remain primarily GEMM-bound.
