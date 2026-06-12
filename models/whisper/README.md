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
(`WHISPER_ENC_H` seam in `cmd/audio/whisper`). See `research/aicpu-whisper`.

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

A100 FFN batching/prewarm update: the A100 hook can also be extended to FC2 with
`WHISPER_A100_FC2=1`, but live measurement is negative (`FC1+FC2` was about
`pass0=48.1s` versus `FC1` at about `44.0s`, with decode/token drift in that
sample), so FC2 remains explicitly experimental. The useful cold-pass win is
prepacking all enabled encoder A100 FFN weights and prewarming the pool at model
load time: FC1 pass0 improved from roughly `43.7s` to `41.1s`, close to native
int8 but still slower than the `~40.1s` baseline on the same sample.

A100 fused FFN experiment: `WHISPER_A100_FFN_FUSED=1` replaces the encoder MLP
block with `FC1(A100) -> FC2(A100)` and fuses GELU into the FC2 activation
quantizer (`QuantizeF32RowsQ8M4GELUInto`) so the 1500x5120 hidden matrix is not
written/read again just for GELU before FC2 packing. The fused packer uses the
same rational `fastTanh` approximation as the normal Whisper GELU and is tested
against explicit GELU followed by normal Q8 M4 packing. On `pod_30.wav` with
`WHISPER_THREADS=6`, the fused path improved encoder time (`encoder+xkv` about
`32.8s` vs baseline `35.6s`) and first-pass wall (`39.2s` vs `40.0s`), but it
also changed decode behavior (`107` tokens vs `73`) and warm-pass wall remained
slower (`37.4s` vs `36.4s`) because decode dominated. Keep it opt-in while
transcript quality and decode-length behavior are studied.

A100 FFN quality diagnostics: `cmd/audio/whisperffndiag` compares per-layer FFN
outputs against the normal native-int8 path at the same pre-MLP input. Example:

```sh
WHISPER_THREADS=6 WHISPER_INT8=1 go run ./cmd/audio/whisperffndiag \
  -model /home/me/models/whisper-turbo/model.safetensors \
  -size turbo -audio /home/me/pod_30.wav
```

It prints layer/variant metrics (`max_abs`, `mean_abs`, `rmse`, `rel_rmse`, and
`cosine`) for `a100_fc1_native_fc2` and `a100_fused_ffn`. On `pod_30.wav`,
`a100_fc1_native_fc2` stays close to baseline (`max_rel_rmse≈0.017`,
`min_cos≈0.99985`), while the fused A100 FFN has a large early-layer drift spike
(`max_rel_rmse≈0.131`, `min_cos≈0.99135`, worst at layer 1). This supports using
A100 FC1 with native-int8 FC2 as the safer next mode and keeping full A100 FC2
opt-in until its quantization is improved.

A100 fused FC2 mode selector: `WHISPER_A100_FFN_FC2_MODE` controls the FC2 side
of `WHISPER_A100_FFN_FUSED=1`. The default `a100` keeps the full A100 fused FFN
and is encoder-fast but changes decode length on `pod_30.wav` (`107` tokens).
The safer `int8` mode runs A100 FC1, applies normal GELU, then uses the native
int8 FC2 path before adding the residual. This preserves baseline token count
(`73` tokens) and transcript shape, but does not improve wall time yet: measured
`pass0=40.1s` / `pass1=38.0s` versus baseline `40.0s` / `36.4s` on the same
sample. It is a quality-safe control path for future tile-level fusion and RVV
activation-packing work.

Tile-level FFN execution: `WHISPER_FFN_TILE_M=N` executes the encoder FFN in row
blocks, so FC1, GELU, FC2, and residual operate on tiles instead of keeping the
full `[seqLen, ffnDim]` hidden buffer live. The tiled path also composes with
`WHISPER_A100_FFN_FUSED=1` and `WHISPER_A100_FFN_FC2_MODE`. Initial
`pod_30.wav` measurements show the scaffold is correctness-safe but not a speed
win yet: native tiles were slower than the full-buffer native path (`tile=256`
about `40.5s` vs baseline `39.8s`), and safe A100-int8 tiles were also slower
(`tile=256` about `41.4s`). Full-A100 tiled mode was closest at `tile=256/512`
(`~39.5–39.6s`) but retained the 107-token decode drift. This confirms the next
bottleneck is activation packing rather than hidden-buffer lifetime alone.

X100 activation prepack for A100 FFN: `WHISPER_A100_X100_PACK=1` moves Q8 M4
activation packing out of the registered A100 worker callbacks and into normal
X100 goroutines before A100 dispatch. This is a measurement/optimization bridge
toward RVV pack kernels. Hardware smoke tests verify that X100-prepacked A blocks
produce byte-equivalent A100 outputs for both plain and GELU-fused packing. On
`pod_30.wav`, full A100 fused FFN improved from about `pass0=39.3s` /
`encoder+xkv=32.9s` / `[a100] ffn=8.74s` to `pass0=37.8–38.0s` /
`encoder+xkv=31.4–31.5s` / `[a100] ffn=7.35–7.49s`. It still changes decode
length (`107` tokens), so it remains opt-in; the safe native-FC2 mode preserves
`73` tokens but is still slower than baseline.

A100 default eligibility decision: keep native int8 as the default. After X100
activation prepack, the fastest A100 path (`WHISPER_A100_FFN_FUSED=1
WHISPER_A100_FFN_FC2_MODE=a100 WHISPER_A100_X100_PACK=1`) is faster on wall time
(`pass0≈38.0s`, `pass1≈36.2s`) but changes decode length (`107` tokens vs the
baseline `73`). The quality-safe hybrid (`WHISPER_A100_FFN_FC2_MODE=int8`) keeps
`73` tokens but remains slower (`pass0≈40.7s`, `pass1≈37.6s`) than baseline
native int8 (`pass0≈39.4s`, `pass1≈36.5s`). A100 should therefore remain opt-in
until FC2 quantization drift is reduced or the safe hybrid beats baseline.

Layer-selective A100 FFN: `WHISPER_A100_FFN_LAYERS` restricts
`WHISPER_A100_FFN_FUSED=1` to a comma/range selector such as `12-31` or
`16-31`. This targets the diagnostic finding that early layers have the largest
full-A100 FFN drift. On `pod_30.wav`, ranges `12-31` and `16-31` preserved the
baseline `73` token output, unlike all-layer/full later-layer-only runs that can
produce `107` tokens. Timing was variable: `16-31` produced fast warm passes
(e.g. `encoder+xkv≈32.9s`, `pass≈37.3s`) but was not stable enough across
repeats to replace native int8 by default. Keep this as an opt-in quality/speed
tradeoff tool while FC2 quantization and activation packing are improved.

Native-compatible A100 Q8 row scaling: `WHISPER_A100_NATIVE_Q8=1` packs A100
Q80x32 weights with one scale per output row and packs GELU activations with one
scale per activation row, repeated across K32 blocks. This mirrors the existing
native int8 quantization contract while still using the A100 tile layout. On
`pod_30.wav`, the full fused A100 FFN with X100 activation prepack and row-scale
Q8 preserved the baseline `73` token output and became faster than native int8:
`pass0≈37.0s` / `pass1≈35.3s` versus baseline `pass0≈40.1s` / `pass1≈37.0s` in
the same run. The older block-scale A100 Q8 path remained slightly faster in
encoder-only terms but drifted to `107` tokens. Recommended opt-in candidate:
`WHISPER_A100_FFN_FUSED=1 WHISPER_A100_X100_PACK=1 WHISPER_A100_NATIVE_Q8=1`.

A100 row-scale confirmation: a repeat-3 confirmation on `pod_30.wav` with
`WHISPER_A100_FFN_FUSED=1 WHISPER_A100_X100_PACK=1 WHISPER_A100_NATIVE_Q8=1`
preserved the baseline `73` token transcript and improved every pass in the same
run: baseline `40.6s/37.2s/36.0s` versus A100 row-scale `37.4s/35.7s/35.5s` for
passes 0/1/2. This supersedes the earlier block-scale default-eligibility
finding: row-scale A100 is now a strong opt-in/default-candidate path, but should
be validated on a broader audio set before becoming the automatic default.
