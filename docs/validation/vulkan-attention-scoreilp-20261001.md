# Vulkan key32 score interleaving — 1 October 2026

Explicit F32 score interleaving reduces matched Go Whisper request medians by 4.82–5.37% on Sigma without changing retained outputs. JFK with native Silero VAD and word timing improves by 4.19%. Serving and encoder defaults are unchanged; the original-engine speed target is still unmet.

## Change and measurements

`NewVkAttentionKey32ScoreILPF32` computes keys `x` and `x+16` together, reusing each query value and maintaining two independent FMA chains. Each key retains increasing-channel reduction order. Softmax, key32 tiles, probability sum, value reduction, tail masking, shared storage and barriers retain the prior key32 contract. Input, shared data, accumulation and output remain F32. No optional Vulkan feature, opcode-envelope expansion, quadratic score buffer, quantisation or fallback is added.

`NewVulkanEncoderTile64Key32ScoreILP` selects the new attention with the existing tile64 projections. The baseline constructor and `NewVulkanEncoderTile64Key32` remain intact. The benchmark backend is `vulkan-f32-tile64-key32-scoreilp`.

The starting revision was `ec7714f02cbf63496898914544a2c4b5e6608c5f`. Go was `go1.26.2 linux/amd64`. Both arms used the pinned Turbo F16 checkpoint widened to F32 (`542566a422ae4f3fd23f1ba11add198fca01bbf82e66e6a2857b3f608b1eb9d1`), four CPU threads, `GOMEMLIMIT=4GiB` and the same input pins/options. Each arm ran five requests in one isolated process. Model loading and Vulkan preparation are excluded from request times and separately recorded in the JSON reports. These are process-local repeated requests, not five independent cold launches.

| Fixture and options | Prior key32 median (s) | Score ILP median (s) | Request time change |
|---|---:|---:|---:|
| JFK, VAD/words off | 7.467 | 7.089 | −5.07% |
| MINDS Portuguese, VAD/words off | 7.437 | 7.079 | −4.82% |
| MINDS French, VAD/words off | 6.910 | 6.539 | −5.37% |
| JFK, native VAD/words/gap preservation on | 8.412 | 8.060 | −4.19% |

All five candidate requests per fixture matched baseline window structures, text, content tokens, segments and times exactly. The VAD/word comparison additionally covers retained speech and audio coordinates, eligibility and every word time. Five-repeat JFK, PT, FR and two-group results matched the previously qualified key32/KV-sharing reports; silence produced no retained windows in all five requests. Only JFK has a fresh matched VAD/word timing pair here. The other VAD fixtures provide regression evidence without a new speed claim. Two-group runs kept the 4 GiB Go heap target under the unchanged 8 GiB hard cap.

## Native operator and graph checks

At the full 1500×20×64 attention shape, the initial interleaved ABBA/BAAB diagnostic measured about 78.4→66.9 ms (about 14.6%). Six samples per arm followed warm-up in one process. All candidate outputs were bit-identical to key32. The retained test enforces bitwise full-shape agreement; five repeated invocations passed, alongside the existing independent F64 scalar cases, odd shapes, extreme sequence tails, softmax stress, sentinels and native-memory cleanup. Scalar tolerances were unchanged (`2e-5` normal and `2e-4` absolute for the existing stress cases).

Two synthetic encoder geometries passed five repetitions, including cancellation, drain, reuse and independent scalar boundary checks. Offline admission rejects cancelled contexts and insufficient shared memory before native construction; encoder tests reject nil/invalid/cancelled inputs and verify that the candidate does not select Q8 or change defaults.

A focused independent read-only delegate review completed. It found no concrete issue in score order, bounds, ownership, default selection or benchmark wiring. It did not execute models or establish independent acoustic accuracy.

## Rejected candidates

The same bounded window tested exact-order output-chain interleaving (+8.46% kernel time), one redundant barrier removed (−0.43%, too small to retain), their combination (+7.74%) and a 32-query workgroup (about +5%). All passed synthetic checks but were not retained. The accepted score-interleaved shader keeps the original barriers and 16-query geometry. Sources and measurements are retained as evidence; diagnostic dispatch changes and external-shader tests were removed.

The original engine's retained Intel startup log reports subgroup width 32, 49152 shared bytes, FP16 support and no matrix cores. No new cooperative-matrix backend or capability support is established by this window.

## Checks, isolation and release

The authorised container retained four-CPU quota, `taskset -c 0-7`, 8 GiB memory/memory-plus-swap cap, Intel ICD/render device, no network and a read-only root. Guards required idle Qwen `/slots` and at least 6 GiB host available memory. Qwen serving, Gemma state, production binaries and profiles were unchanged.

Verified checks:

- affected tests, new admission tests repeated ten times, and vet;
- `make model-layout-check host-build host-vet host-test docs-check`;
- whole-tree `go test -race -p=2 -count=1 -timeout=180s ./...`;
- Linux ARM64 and RISC-V whole-tree builds, compilation only;
- six shader-checker tests and all 27 stored/rebuilt SPIR-V modules validated and normalised-matched offline.

Two outer shell calls timed out while their isolated containers continued. The first container's exit 0 alone did not establish whole-tree race completion: its attached log stopped early. A separate confirmation run has `PASS_RACE`, `PASS_ARM64_BUILD` and `PASS_RISCV64_BUILD` in the complete `podman logs`, with exit 0. Its attachment timed out just before finishing; logs/state were recovered and the stopped container removed. All failures, empty benchmark invocations caused by an incorrect test filter, partial logs and completion records are retained; empty invocations supply no benchmark evidence.

At 05:35 UTC, native/build work and candidate containers were drained. Only pre-existing `wrdp` containers remained, no Go process was running, and Qwen slot 0 was idle. `@llama` received explicit release of the experiment/build/restart hold.

## Reproduction and evidence

[Evidence directory](../../benchmarks/speech-foundations/vulkan-attention-scoreilp-20261001/) contains pinned reports, matched and historical regression references, rejected GLSL/SPIR-V, native logs, guarded runner/environment files, resource samples, partial and recovered completion logs, and static shader checks. `comparison.json` records exact-output checks and timing medians. The collection helper is retained as source text. Hashes identify inputs/artifacts, not acoustic quality.

Build test binaries from this revision. Set `GO_PHERENCE_TEST_ATTENTION_SCOREILP=1` for `TestVulkanNativeAttentionScoreILP`, and `GO_PHERENCE_TEST_VULKAN_ENCODER_SCOREILP=1` for `TestVulkanEncoderScoreILPNative`; both require a named physical device and a timeout no longer than 120 seconds. Trained arms use `TestWhisperPerformanceGoalArm` with the archived environment files and guarded runner. Ordinary tests skip native work and establish no hardware evidence. Use new report paths; existing reports are protected by exclusive creation.

Original Q5/F16 execution uses different precision and timing boundaries. The 7.089 s Go JFK result still does not meet the retained original Q5 3.350 s or F16 6.183 s process medians. Independent acoustic/word-timing, long-form and resume acceptance remain open.
