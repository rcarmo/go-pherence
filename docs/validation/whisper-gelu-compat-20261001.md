# Original-style encoder GELU in padded-key mode — 1 October 2026

A test-only Vulkan shader reproduces the original GGML tanh-form GELU bit-for-bit on the original's captured layer-0 FC1 input. Substituting it into the explicit padded-key encoder leaves every tested transcript, segment and word output unchanged, including the Portuguese endpoint miss. Request costs are tied. No runtime option is added; defaults, decoder arithmetic, K/V storage and services are unchanged.

## Diagnostic operator

The shader keeps the original expression order: `0.5*x*(2-2/(exp(2*v)+1))`, with `v=sqrt(2/pi)*x*(1+0.044715*x*x)`. It does not substitute erf, clamp inputs or add a further approximation. Its SPIR-V is pinned by SHA-256 `08021761…e340` and admitted only through test environment variables. K/V F16 storage rounding is deliberately excluded from this trial.

Analytic checks cover 120,009 inputs, extremes, tails and in-place execution. On the original engine's captured FC1 output for the cropped JFK fixture (`op8-fc1`, 7,680,000 values), five repeats match the original's GELU output (`op9-gelu`) in every F32 bit. Pre-cancellation, fresh reuse and allocation guards pass. This verifies the operator on that tensor only.

## Model results

The candidate replaces only the encoder FFN activation in `vulkan-original-q5-padded-keyextent`. Baseline and candidate use the same executable, model and input pins, CPU4/heap4GiB, parallel decoder rows and four projection workers. Five repeats include the first request; arms run sequentially without randomisation.

| Fixture | Padded erf median (s) | Padded original-GELU median (s) | Change | Outputs |
|---|---:|---:|---:|---|
| JFK noVAD | 6.133 | 6.123 | −0.16% | identical |
| PT row0 noVAD | 6.103 | 6.096 | −0.10% | identical |
| FR row0 noVAD | 5.691 | 5.696 | +0.09% | identical |
| JFK VAD + words | 6.873 | 6.848 | −0.36% | identical |
| Two JFK groups, VAD + words | 13.856 | 13.837 | −0.14% | identical |

All arms repeat their own outputs exactly across five requests, and encoder stats are unchanged. The expanded PT row1, PT row2, podcast noVAD, podcast VAD+words and silence fixtures also match the archived padded-erf outputs exactly in all five repeats.

Consequently the PT row0 endpoint stays 7.38s against the original's 7.36s; the PT2 late endpoint and podcast segmentation differences reported for the [padded-key mode](whisper-padded-keyextent-20261001.md) remain. GELU form is not the cause of those mismatches. The timing differences are within run-to-run noise and do not count as a speed result.

## Decision and verification

Rejected as a runtime mode: it produces no output change on the tested fixtures and no measurable speed change. The verified operator expression, its common-input evidence and the candidate patch are archived for later combined-arithmetic studies.

The restored tree passes `make model-layout-check host-build host-vet host-test docs-check` (536 Markdown files, zero broken links), opt-in race tests for `backends/vulkan` and `model/whisper`, and ARM64/RISC-V cross-builds. All native runs executed under the @llama isolation hold with CPU4/8GiB/no-swap, physical Intel GPU, Qwen idle, host available memory ≥6GiB and ≤120s native deadlines. The final silence run exited 0 without OOM; an outer command timeout interrupted runner cleanup, so its state and log were recovered and the stopped container removed before the final gates.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-gelu-compat-20261001/) retains baseline/candidate/expanded JSON and environment files, native logs/states/exits/monitors, the shader source/SPIR-V, harness and patch text, parser/comparison, provenance and gate logs. Captured F32 state arrays and test binaries are recorded by hash only.

Open: K/V F16 storage-rounding compatibility, the PT/podcast boundary causes, independent acoustic/word timing, long-form/resume/fault qualification, and the overall original-speed target.
