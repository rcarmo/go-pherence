# Original decoder compatibility: padded cross-attention and tanh GELU — 1 October 2026

A CPU emulation of the original decoder's arithmetic boundaries, checked against swaps with the original decoder itself, shows two decision-relevant differences in the Go decoder:

1. **Cross-attention key extent.** whisper.cpp pads the encoder context to a multiple of 256 (1500→1536) and passes no mask to flash attention, so 36 zero keys enter every decoder cross-attention softmax.
2. **MLP activation.** ggml's `gelu` is the tanh form; the Go decoder used erf GELU.

The new explicit option `PCMTranscribeOptions.OriginalDecoderCompatibility` applies both. With the integer-dot MMQ encoder it moves Portuguese row 0 from 9.04 s to 7.38 s: the original's no-F16 result, 20 ms from its default 7.36 s. JFK still ends at 11.00 s against 10.40 s; that difference is attributed to the encoder. Defaults are unchanged and request cost is unchanged.

## Emulation and swaps

A test-only CPU decoder reimplemented each step with independently switchable original boundaries: padded cross extent, F16 self/cross K/V storage, Q8_1×Q5_0 FC2, Q8_1 for every projection in the three-token prompt step, and tanh GELU. It decoded with the production timestamp rules from saved encoder states. With no switches it reproduced the Go decoder's result in each case: 9.04 s for PT from the original no-F16 hidden states, 10.40 s for JFK.

| Fixture | Hidden fed in | Decoder | End |
|---|---|---|---:|
| PT row 0 | original no-F16 | original | 7.38 |
| PT row 0 | original no-F16 | Go / emulation base | 9.04 |
| PT row 0 | original no-F16 | emulation, pad only | **7.38** |
| PT row 0 | original no-F16 | emulation, tanh only | **7.38** |
| PT row 0 | original no-F16 | emulation, F16 K/V only / Q8_1 FC2 only | 9.04 |
| PT row 0 | Go integer-dot all | original | 7.38 |
| PT row 0 | Go integer-dot all | emulation, pad only / tanh only / F16 K/V / FC2 / prompt | 9.04 |
| PT row 0 | Go integer-dot all | emulation, **pad + tanh** (any added F16/Q8 switches) | **7.38** |
| PT row 0 | Go exact / original default | emulation base and pad | 7.38 |
| JFK | original no-F16 | original | 10.40 |
| JFK | original no-F16 | emulation base / pad+tanh / all | 10.40 |
| JFK | Go integer-dot all | original | **11.00** |
| JFK | Go integer-dot all | emulation base / pad+tanh / all | 11.00 |

F16 caches and Q8_1 decoder activations are not needed for these decisions. JFK's 11.00 s comes from the encoder: the original decoder gives the same 11.00 s from Go integer-dot hidden states. The Go integer-dot encoder is still closer to the original than the exact encoder (JFK RMSE 0.068 versus 0.086 against original no-F16). Encoder GELU form and F16 K/V storage are the next encoder candidates.

## Implementation

- `DecoderState` gains private `crossPadKeys` and `tanhGELU`, set only through `applyOriginalDecoderCompatibility(encLen)`.
- `crossAttentionHeadMajorPadded` adds virtual zero keys. Their exact-zero logits join the softmax maximum and denominator, and they contribute no value. The alignment observer sees the real-key probabilities. The existing function forwards with zero padding and keeps its old `softmax` path. Padded states never take the optional CUDA attention path.
- `geluOriginalTanh` uses ggml's operation order, `0.5*x*(2-2/(exp(2v)+1))`.
- `decodePCMWindow` applies the option to the main state and copies it to the word-alignment state. Language detection is unchanged.
- The benchmark arm sets the option through `GO_PHERENCE_WHISPER_BENCH_ORIGINAL_DECODER=1` and records it in JSON.

Unit tests cover the pad count (multiple of 256), exact equivalence to `softmax` when padding is zero, an analytic padded denominator, the all-negative case, tanh GELU against a float64 reference, and default-off state/options. With the option off, a native PT run returns output byte-identical to the pre-change MMQ run (9.04 s).

## Fixture results (`vulkan-original-q5-padded-integer-dot-mmq` + compatibility, 5 repeats)

All arms repeat their outputs exactly. Original references come from different harnesses, listed per row. Only PT/JFK noVAD use the same reusable-context harness and request boundary.

| Fixture | Original | Go all-intdot | Go all-intdot + compatibility | Median request |
|---|---|---|---|---:|
| PT row 0 (same harness) | 0–7.36 (no-F16 7.38) | 0–9.04 | **0–7.38** | 4.366 s |
| JFK (same harness) | 0–10.40 | 0–11.00 | 0–11.00 | 4.400 s |
| FR row 0 | — | 0–2.42 | 0–2.42 | 4.036 s |
| JFK VAD + words | 2 segments (original compaction) | 1 segment, no tail | 1 segment, no tail | — |
| Two JFK groups VAD + words | 2 | 2 | 2; text equal | — |
| PT1 (whisper-cli) | 0–4.58 | 0–4.69 | 0–4.69 | — |
| PT2 (keyextent original) | last 30.00–35.12 | 30.00–43.76 | 30.00–43.76 | — |
| Podcast noVAD | 5 segments | 6 | 6 | — |
| Podcast VAD + words | 7 segments | 6 | **7**, text equal | — |
| Silence | — | none | none | — |

The PT request median is 4.366 s with compatibility against 4.405–4.420 s without, so the option has no measurable cost. Overall speed is unchanged against the original's 3.0 s.

## Verification and isolation

`make model-layout-check host-build host-vet host-test docs-check` (539 Markdown files, zero broken links), race for `model/whisper`, and ARM64/RISC-V builds pass. The temporary emulation and state tests were removed from the tree after use and are kept as text in the evidence. One diagnostic run exited 1 because the state test first used a widened-source constructor on a packed-only model; its rerun passed.

CPU-only emulation and the GPU runs used separate @llama isolation holds under the usual guards (CPU4/8GiB/no-swap, Qwen idle, host ≥6 GiB, ≤120 s). All containers exited without OOM and were removed.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-decoder-compat-20261001/) holds the benchmark JSON/environments, emulation and swap logs, comparison script and output, gate logs and provenance.

## Next

1. JFK encoder attribution: test encoder tanh GELU and F16 K/V storage with the integer-dot encoder against the 10.40 s decision.
2. A GPU decoder, about 0.7 s of the remaining 1.4 s gap. It must keep these compatibility semantics.
