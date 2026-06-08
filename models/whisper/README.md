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

The reusable DSP and GEMM kernels (mel/FFT, int8 matmul, experimental FP16/Zvfh)
live in `backends/simd/fft` and `backends/spacemit`; the files here are
Whisper-specific orchestration and quantization tuned for byte-exact transcripts.

Production sub-1.0 RTF path: EP-encoder + Go int8 turbo-decoder hybrid
(`WHISPER_ENC_H` seam in `cmd/audio/whisper`). See `research/npu-whisper`.

Native K3 path status: `WHISPER_INT8=1` enables the full native IME/RVV int8
encoder+turbo decoder. `WHISPER_FP16_ATTN=1` is an experimental X100 RVV/Zvfh
attention path (QKᵀ and softmax·V in FP16 with F32 accumulation). On `pod_30.wav`,
FP16 attention is transcript-safe but currently slower than int8 attention
because many small per-head GEMMs dominate after F32→F16 packing was vectorized.
`WHISPER_FP16_HEAD_BATCH=1` enables an additional head-batched FP16 experiment;
it reduces summed GEMM fanout time but is slower end-to-end today because the
larger batched working set increases copy/allocation/softmax overhead. Per-head
FP16 GEMMs deliberately keep row-parallel WHISPER_THREADS fanout: attention
shape benchmarks (1500x1504x64) show 6 threads at ~4.3 ms vs 1 thread at ~23 ms.

A100 int8 experiment: `WHISPER_A100_FC1=1` routes the large-v3/turbo encoder
FFN expansion (`1280 -> 5120`) through the native A100 Q8_0 x Q8_0 N32/K32
`K3I8I8` path via `aipool` (`/proc/set_ai_thread`, cores 8-13 by default).
The opt-in path defaults to 6 A100 workers and disables generic `IME2_TCM_ACT`
staging unless explicitly requested; the current FC1 integration performs its own
activation packing and TCM staging did not improve whole-pass timings in the
worker-count grid. The path is real A100 execution and currently opt-in only: on `pod_30.wav` with
`WHISPER_THREADS=6`, it measured `pass0=43.9s` / `encoder+xkv=38.8s` versus the
baseline native int8 `pass0=40.2s` / `encoder+xkv=35.8s`. The FC1 kernel itself
runs (~2.7s total), but the extra activation/weight Q8_0 packing and scheduler
interaction still erase the win. Keep this as infrastructure while improving
packing/pooling before making it default.

Warm-cache A100 note: with `WHISPER_REPEAT=2`, A100 FC1 improves to
`pass1=37.8s`, but baseline native int8 is still faster at `pass1=35.7s`. The
FC1 kernel itself is not the bottleneck; activation packing and scheduler/pool
integration remain the overhead to remove before enabling A100 by default.
Current overhead reductions implemented: the A100 helper reuses one Q8 activation
pack scratch per worker, and Whisper no longer clones/pads the full activation
matrix just to satisfy the M4 kernel tail; non-M4 tails are handled inside the
A100 pooled helper with small per-worker scratch.
