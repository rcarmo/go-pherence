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

## Readiness

Native inference is not yet ready. The package currently pins and validates the exact architecture. Readiness requires all plan gates: frontend, encoder, adaptor, insertion, Qwen decoder, tokenizer/processor, transcript parser, selected logits, and end-to-end real-audio parity.
