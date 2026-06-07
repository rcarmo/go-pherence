# models/whisper

Whisper large-v3 speech-to-text: encoder, decoder, and the int8/turbo decode path
tuned for the SpaceMIT K3 (MilkV Jupiter 2).

| Area | Files |
|---|---|
| Entry / config / load | `whisper.go`, `config.go`, `load.go`, `language.go` |
| Encoder | `encoder.go`, `gpu_encoder.go` |
| Decoder | `decoder.go`, `decode.go`, `decoder_batch.go`, `decoder_bufs.go`, `batched.go`, `chunked.go` |
| Speculative (turbo) decode | `speculative.go`, `speculative_accept.go` |
| int8 / quant | `int8_linear.go`, `linear_opt.go`, `turboquant.go` (compressed KV cache) |
| Tokenizer / output | `tokenizer.go`, `align.go`, `vtt.go` |

The reusable DSP and GEMM kernels (mel/FFT, int8 matmul) live in
`backends/simd/fft` and `backends/spacemit`; the files here are Whisper-specific
orchestration and quantization tuned for byte-exact transcripts.

Production sub-1.0 RTF path: EP-encoder + Go int8 turbo-decoder hybrid
(`WHISPER_ENC_H` seam in `cmd/audio/whisper`). See `research/npu-whisper`.
