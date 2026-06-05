# Whisper on RISC-V (SpaceMIT K1/K3) — RVV + IME optimization

This document describes the optimization of the Whisper speech-to-text pipeline
(`cmd/whisper`, `models/whisper`) for the SpaceMIT K1/K3 SoC as found on the
**MilkV Jupiter 2** (8× X60 RISC-V cores, RVV 1.0 vector unit, and the **IME**
integer matrix engine reached via the `vmadot` instruction).

The end result on `whisper-large-v3` is **~17× faster** warm inference than the
naive single-core F32 baseline, with a **byte-identical transcript**, on a
RISC-V board with no GPU.

## Results (large-v3, jfk.wav 11s, warm/resident inference)

| Stage | Warm inference | RTF |
|-------|---------------:|----:|
| Baseline (F32, 1 core)                         | ~310 s | ~28 |
| F32 fully optimized (RVV m4, frame-tiling, GEMM attention/conv, threading) | ~61 s | ~5.5 |
| + int8 IME encoder linears + RVV quantization  | ~37 s | ~3.3 |
| + int8 attention + int8 decode + fast GELU     | ~22 s | ~2.0 |
| + 6 threads                                    | ~19 s | ~1.7 |

RTF = real-time factor (wall seconds per second of audio). "Warm" is the
steady-state resident cost after the one-time weight quantization/packing has
been cached — the relevant number for an always-on transcription endpoint.

## Techniques

The optimization proceeded from F32 SIMD through to the integer matrix engine.
The high-value levers, in order of impact:

1. **RVV `m4` dot kernel.** The vector FMA throughput saturates at LMUL=4
   (~12 GFLOP/s); `m1`/`m8`/`m4×2` all measured slower, so `m4` is the proven
   F32 ceiling. Wired into the Sdot/SGEMM/norm kernels under
   `backends/simd/runtime`.
2. **Frame-tiling the encoder GEMM** (`WHISPER_BLOCKM`). The encoder linear is
   memory-bandwidth bound, not FLOP bound; tiling the frame (row) dimension keeps
   activations resident in L2 across the output sweep.
3. **GEMM-reformulated attention and conv stem.** The conv stem became
   im2col + RVV GEMM (≈80× on that phase); attention became batched RVV GEMMs.
4. **Threading** across the independent frame/head/output dimensions, capped at
   4 by default (see *Threads & thermals*).
5. **int8 IME encoder linears** (`WHISPER_INT8`). Per-row symmetric int8
   quantization feeds the `vmadot` integer matrix engine (`backends/spacemit/ime2`),
   which is a separate datapath from the F32 FMA unit and moves 4× less weight
   bandwidth. ~2.2× over F32 at 4 threads, ~0.9% mean quantization error, and the
   greedy-decoded transcript is unchanged. Weights are quantized + tile-packed
   once and cached, so the cost amortizes to zero on a resident server.
6. **RVV activation/weight quantization** (`FindMaxAbsRVV` + `QuantizeF32ToI8RVV`).
   Replaces the scalar quantize loop; cuts per-forward quant ~7× and speeds the
   one-time weight packing.
7. **int8 IME attention.** Both per-head GEMMs (`Qh@Khᵀ` and `scores@Vh`) run on
   the IME with the softmax in between. Validated byte-identical; ~1.85×.
8. **int8 decode.** The decoder streams its entire ~2.9 GB weight set from RAM
   *per token* — it is bandwidth bound — so int8 weights cut decode ~1.7×.
9. **Fast GELU.** A float32 Padé[7/6] `tanh` approximation (error < 1e-4)
   replaces ~246M libm `float64` `tanh` calls per forward, cutting the GELU/LN
   "other" phase ~3.7×.

## Runtime tunables

| Env var | Default | Meaning |
|---------|---------|---------|
| `WHISPER_INT8`        | off  | Enable the full int8 IME pipeline (encoder linears, attention, decode). |
| `WHISPER_INT8_NOATTN` | off  | Keep attention in F32 while int8 is on (accuracy debugging). |
| `WHISPER_THREADS`     | min(GOMAXPROCS, 4) | Worker threads for the batched GEMMs. |
| `WHISPER_BLOCKM`      | 32   | Frame-tile height for the encoder GEMM. |
| `WHISPER_REPEAT`      | 1    | Run N transcription passes (measure warm/resident cost). |
| `WHISPER_DEBUG`       | off  | Print per-phase timing to stderr. |

Recommended resident invocation:

```sh
WHISPER_INT8=1 WHISPER_THREADS=4 bin/whisper-k3 \
  -model models/whisper/whisper-large-v3.safetensors -size large-v3 \
  -audio jfk16.wav
```

## Threads & thermals

The board **brown-out reboots** under sustained all-8-core RVV + IME load — a
power/thermal limit, not a software fault. The default thread cap is therefore
**4**. `WHISPER_THREADS=6` is faster (~19 s vs ~22 s warm) and survived testing,
but it runs closer to the edge; use it only with cooling headroom. 8 reliably
resets the board.

## Building

```sh
make whisper       # native build -> bin/whisper
make whisper-k3    # riscv64 build (RVV+IME) -> bin/whisper-k3
```

The RVV and IME kernels are gated by `//go:build riscv64` and selected at runtime
via CPU feature detection (`HasRVV`, `HasSGEMM`), so a native riscv64 build picks
them up automatically; no special build tags are required. `whisper-k3` forces
`GOARCH=riscv64` (with `CGO_ENABLED=0`) so it can also be cross-compiled from an
x86 host.

## Diarization (speaker labels)

`cmd/whisper -timestamps -diarize` produces multi-speaker transcripts using an
**ECAPA-TDNN** speaker-embedding network (`models/speaker`, SpeechBrain
`spkrec-ecapa-voxceleb` topology, parity-validated to cosine ≈0.9999 vs upstream)
on top of the optimized Whisper path: energy VAD → ECAPA embeddings per segment →
agglomerative clustering → singleton-label smoothing → speaker-tagged cues.

```sh
# One-time: fetch + convert the speaker weights (torch-free; python3 + numpy only)
make speaker-weights      # -> models/speaker-ecapa-voxceleb.safetensors

# Transcribe with speaker labels
WHISPER_INT8=1 bin/whisper-k3 -model models/whisper/whisper-large-v3.safetensors \
  -size large-v3 -timestamps -diarize -audio meeting.wav
# [0.00 - 5.82] Speaker 1: ...
# [14.30 - 18.08] Speaker 2: ...
```

Flags: `-speaker-model` (default `models/speaker-ecapa-voxceleb.safetensors`) and
`-speaker-threshold` (default 0.3, lower = more speakers). If the speaker model is
absent the command logs a warning and falls back to single-speaker labels so
transcription still succeeds. `cmd/speakercheck` runs the speaker path alone
(VAD + embeddings + clustering) for tuning without Whisper.

The same RVV/SIMD kernels accelerate ECAPA: its 1×1 pointwise convolutions and
its kernel>1 convolutions (Conv0, Res2Net) both route through im2col + RVV
SGEMM, and segments are embedded in parallel across cores (`SPEAKER_THREADS`,
default min(GOMAXPROCS, 6)). On a 30s 2-speaker podcast window the speaker pass
dropped from ~93s to ~12s. Weights are converted from the SpeechBrain
`.ckpt` (a torch zip) by `scripts/ckpt_to_safetensors_numpy.py`, which reads the
checkpoint with only numpy + the stdlib — no torch — so the conversion runs on
the board itself.

> WAV decoding was also rewritten from a per-sample reflection-based
> `binary.Read` loop to a single buffered read + direct little-endian decode
> (`loader/audio/wav.go`): loading a 26-minute clip went from ~37s to ~0.17s.

## Validation status

The int8 pipeline is validated **byte-identical to the F32 path on jfk.wav**.
Per-row int8 quantization carries ~0.9% mean / ~6% worst-case error per layer,
which the greedy-argmax decode tolerates here, but a broader multi-clip
validation pass is advisable before making `WHISPER_INT8` the unconditional
default.
