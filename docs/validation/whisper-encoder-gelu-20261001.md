# Original encoder GELU with integer-dot MMQ — 1 October 2026

With the integer-dot MMQ encoder, the encoder GELU form decides JFK's segment end. The new private backend `vulkan-original-q5-padded-integer-dot-mmq-tanh` uses the original GGML Vulkan tanh-form GELU in the encoder stem and FFN. Combined with `OriginalDecoderCompatibility`, it reproduces the original's same-harness results exactly: JFK 0–10.40 and PT 0–7.36, with identical text. Request cost is tied with the erf-GELU MMQ backend.

This reverses the [earlier GELU trial](whisper-gelu-compat-20261001.md) only for this backend. That trial used the F32 padded-key encoder, where outputs did not change. Under Q5×Q8_1 projections the encoder sits on a different precision boundary.

PT2's second 30 s window also needs the original's whole-clip mel floor and previous-text prompt. Both are attributed below and are not implemented here.

## Encoder attribution (test-only)

A test-only plan factory substituted pinned shaders into the MMQ encoder's plans: the original tanh GELU, or bitwise round-to-nearest-even F16 storage rounding of K and V just before attention. The latter emulates whisper.cpp's F16 `kv_pad` copy; a CPU oracle checked it bit-exact against `Float16Array` on 2.4 M values, including subnormal and overflow edges. The factory failed closed unless it found exactly the expected GELU and attention stages in each plan. Hidden RMSE is against the original no-F16 encoder output; decoding uses the Go decoder with compatibility on.

| Encoder arm | JFK RMSE | JFK end | PT RMSE | PT end |
|---|---:|---:|---:|---:|
| base (erf GELU) | 0.0683 | 11.00 | 0.0698 | 7.38 |
| tanh GELU | **0.0491** | **10.40** | **0.0494** | 7.38 |
| F16 K/V | 0.0706 | 10.40 | 0.0696 | 7.36 |
| tanh + F16 K/V | 0.0521 | 10.40 | 0.0495 | 7.38 |

Original: JFK 10.40, PT 7.36 (no-F16 7.38). Tanh GELU is the original operator and cuts the distance by about 28% on both clips. F16 K/V storage moves the marginal decisions without reducing the distance, so it was not adopted.

## Implementation

- `backends/vulkan/shaders/gelu_original_tanh_f32.glsl` uses whisper.cpp's `op_gelu` expression and operation order: `0.5*x*(2-2/(exp(2v)+1))`. Its instruction stream matches the diagnostic shader that an earlier trial validated bit-exact against original GGML outputs on 7.68 M inputs. `NewVkGELUOriginalTanhF32` reuses the existing GELU operator ABI and admission.
- `vulkanLinearOriginalQ5PaddedIntegerDotMMQTanh` equals MMQ apart from the GELU constructor. It is packed-only, needs explicit integer-dot enablement and is selected only by the benchmark/profile backend name. Defaults are unchanged.
- Tests:
  - an offline float32 model against a float64 tanh reference (max abs 3.7e-7);
  - an SPIR-V contract test that admits only `Exp` and rejects `Tanh`;
  - cancellation and limit tests;
  - an explicit native test against the model on 7.68 M samples, with tail sizes and exact in-place alias. On Intel Iris Xe it measured max abs 6.3e-7, 87.6% bit-equal to the libm-rounded model and alias outputs identical.
- The offline shader gate now covers 33 shaders, all passing; the new shader rebuilds byte-identically.

## Fixtures (5 repeats each, outputs repeat exactly)

| Fixture | Original | MMQ + decoder compat | **MMQ-tanh + decoder compat** |
|---|---|---|---|
| JFK (same harness) | 0–10.40 | 0–11.00 | **0–10.40** |
| PT row 0 (same harness) | 0–7.36 | 0–7.38 | **0–7.36** |
| FR row 0 | — | 0–2.42 | 0–2.42 |
| JFK VAD + words | 2 segments (compaction) | 1 | 1 |
| Two JFK groups VAD + words | 2 | 2, text equal | 2, text equal |
| PT1 (whisper-cli) | 0–4.58 | 0–4.69 | 0–4.69 |
| PT2 | last 30.00–35.12 | 30.00–43.76 | 30.00–43.76 |
| Podcast VAD + words | 7 | 7, text equal | 7, text equal |
| Silence | — | none | none |

Median request times, MMQ-tanh against MMQ (both with decoder compatibility): JFK 4.403 vs 4.400 s, PT 4.380 vs 4.366 s, FR 4.015 vs 4.036 s. Tied.

## PT2 second-window attribution

The original's window-2 tokens are `<|0.00|> E aí <|5.12|>`; Go emits `<|13.76|>`. Diagnostics on that window:

| Window-2 condition | Hidden RMSE vs original no-F16 | End |
|---|---:|---:|
| Go per-window mel | 0.361 | 13.76 |
| + previous-text prompt | 0.361 | 13.76 |
| + whole-clip mel floor | 0.060 | 13.76 |
| **+ floor + previous-text prompt** | 0.060 | **5.12** |
| original hidden + previous-text prompt | — | 5.12 |
| original hidden, any decoder-emulation arm, no prompt | — | 13.76 |

Two whisper.cpp behaviours, read from source, explain it:

- `log_mel_spectrogram` clamps at the **whole-clip** max − 8. Go normalises each window separately, so the quiet second window (99% of its values lie below the clip floor) gets a lower floor.
- `whisper_full` prompts each later window with `<|startofprev|>` plus up to 223 previous decoded tokens, timestamps included. `no_context` clears history only at the start.

A fixed 22.14 s seek was ruled out: Go then decodes "Amém." from 0.00, so the original's second window does start at 30.00.

## Verification and isolation

The model and backend packages, vet, the shader script test and the 33-shader offline gate pass. Full-tree gates run with the multi-window change. The test-only factory, F16 rounding shader and window harnesses were removed from the tree; they are archived as text. Five early diagnostic runs exited 1 on the test's own SPIR-V admission (SPIR-V 1.3, then unadmitted vector/half-pack instructions). The pure-integer rewrite passed on rerun.

All native runs used separate user-authorised @llama isolation windows (CPU4/8GiB/no-swap, physical Intel GPU, Qwen idle, ≤120 s, sequential). Every container exited without OOM and was removed.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-encoder-gelu-20261001/) holds JSON/environments, logs, diagnostic sources/shaders, the shader-gate report, the original token dumps and provenance.
