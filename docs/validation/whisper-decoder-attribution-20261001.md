# Integer-dot timestamp attribution: decoder, not encoder — 1 October 2026

The Portuguese timestamp regression of the integer-dot modes (row-0 end 9.04 s versus the original's 7.36 s) comes from the **Go decoder**, not the integer-dot encoder or F16 attention. Fed the Go integer-dot encoder output, the pinned original decoder ends the segment at 7.38 s. Fed the original's own no-F16 encoder output, the Go decoder ends it at 9.04 s. Runtime and defaults are unchanged.

## Original F16 ablation

Pinned whisper.cpp reusable-context runner, five requests per arm, Vulkan with flash attention, Portuguese row 0 and JFK, no VAD:

| Original arm | PT end | PT request median | JFK end | JFK request median |
|---|---:|---:|---:|---:|
| Default (F16 flash attention) | 7.36 s | 3.000 s | 10.40 s | 3.017 s |
| `GGML_VK_DISABLE_F16=1` (F32 attention, integer-dot MMQ retained) | 7.38 s | **2.874 s** | 10.40 s | **2.891 s** |

Encoder flash attention runs at `GGML_PREC_DEFAULT`; whisper.cpp does not call `ggml_flash_attn_ext_set_prec`. On this device that means F16 K/V, F16-rounded pre-scaled Q, and F16 score/output accumulation (`flash_attn.comp`). Disabling F16 changes the PT boundary by only 20 ms and makes the original about 4% faster on these fixtures. F16 attention therefore explains neither the 9.04 s decision nor the original's speed advantage.

## Encoder states

The pinned diagnostic harness exported the original's mel and final encoder hidden states for PT row 0. The Go encoders consumed the same mel.

| Hidden source | RMSE vs original default | RMSE vs original no-F16 |
|---|---:|---:|
| Original no-F16 | 0.0727 | — |
| Go padded exact F32 | 0.0910 | 0.0921 |
| Go padded integer-dot FFN | 0.0765 | 0.0759 |
| **Go padded integer-dot, all projections** | 0.0754 | **0.0698** |

The Go integer-dot encoder is closer to the original than the exact Go path.

## Decoder swaps

| Decoder | Hidden fed in | PT end |
|---|---|---:|
| Original (`WHISPER_DIAGNOSTIC_HIDDEN_IN`, whisper_full, 5 requests) | own default | 7.36 s |
| Original | original no-F16 / Go exact / Go integer-dot FFN / **Go integer-dot all** | 7.38 s |
| Go CPU decoder | original default / Go exact | 7.38 s |
| Go CPU decoder | **original no-F16** / Go integer-dot all | **9.04 s** |

Every original-decoder run is stable across five requests. Text is the same in all arms.

## Original decoder arithmetic (source, `c44b60b8`)

`ggml_vk_should_use_mmvq` on Intel Linux for Q5_0:

- Batched steps (n>1, the three-token prompt) use MMVQ with Q8_1 activations for every projection.
- Single-token steps with K<2048 (Q/K/V/O, FC1, cross-Q) use the dequantising `MUL_MAT_VEC`.
- FC2 (K=5120) uses MMVQ with Q8_1 activations.
- Self and cross K/V caches are F16; decoder attention is flash attention at default precision.

The Go decoder is exact F32 with dequantised weights. On boundary-sensitive timestamp decisions it can diverge from the original even given identical encoder states.

## Consequence

A GPU decoder that follows the original's arithmetic is now both the largest remaining speed item (Go CPU decoder about 0.7 s per JFK request; original about 9 ms per token on GPU) and the route to original-compatible timestamps. Integer-dot encoder modes remain experimental until that decoder is qualified; their timing differences are no longer attributed to the encoder.

## Evidence and isolation

[Hashed evidence](../../benchmarks/speech-foundations/whisper-decoder-attribution-20261001/) holds the original runs, swap reports, environments, harness sources (pinned original diagnostic include, Go state test as text), logs/states/exits and provenance. Binaries, PCM, mel and hidden arrays are recorded by hash only. The Go diagnostic test was removed from the tree after the run.

Native runs used the @llama isolation hold: CPU4/8GiB/no-swap, physical Intel Iris Xe, Qwen idle, host ≥6 GiB, ≤120 s deadlines. The first swap-harness build exited 1 (missing `WHISPER_VERSION` define); the rebuild with that define passed. Every other container exited 0. None hit OOM, and all were removed.
