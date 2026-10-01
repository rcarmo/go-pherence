# Native Silero bounded parity — 1 October 2026

Native Go Silero passes nine independent, bounded whisper.cpp comparisons with ten repetitions each. The checked Whisper VAD integration still fails on a word crossing a removed JFK gap; it is experimental and unavailable as a serving default.

## Reference and arithmetic

The oracle calls the public VAD API from whisper.cpp `c44b60b8053bbf2a5c1e014f11323fb3f2485177`, built with the retained header/library. [Oracle source](../../scripts/silero-vad-reference.cpp) runs one CPU thread with GPU disabled. [Evidence](../../benchmarks/speech-foundations/native-silero-parity-20261001/evidence.json) pins its binary, model, header, CPU library, input and output hashes. Raw probabilities and test logs are retained beside the evidence file.

Model: Silero 6.2.0, SHA-256 `2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987`, 885,098 bytes. The reference uses 512-sample frames with reflect-64 padding. PyTorch's 576-sample schedule has not been tested.

The maximum/mean absolute probability limits were fixed at `1e-4`/`1e-5` before the first JFK test. They were unchanged after padded JFK failed at maximum error `1.66595e-4`. They are qualification limits for these fixtures, not a universal error bound.

The existing generic two-chain SIMD dot reduction differed from the reference's four eight-lane FMA chains. The separate [Sdot32 primitive](../../backends/simd/runtime/sdot32.go) preserves the reference's tree and scalar-tail arithmetic. Its amd64 assembly is checked bit-for-bit against an explicit scalar tree, including tails, offset slices, aliasing, a float32 FMA midpoint, fallback dispatch and NaN propagation. Other architectures compile with the scalar tree; trained native ARM64/RISC-V execution is untested.

The first JFK span test also used the wrong export conversion: whisper.cpp rounds samples to nearest centisecond, with positive ties up. The test now uses that conversion. Segmentation uses strict `>` minimum-duration admission and the padded frame length for its EOF decision. Retained native spans clip to actual PCM; the oracle's padded EOF is clipped for comparison. The API exposes no invented padded audio.

## Results

| Fixture | Scores | Maximum absolute error | Mean absolute error |
|---|---:|---:|---:|
| JFK | 344 | 3.87430e-6 | 1.83398e-8 |
| Five-second silence | 157 | 5.50877e-7 | 7.97853e-9 |
| JFK gain 0.1 | 344 | 1.14441e-5 | 4.86511e-8 |
| JFK with 15-second silence before/after | 1,282 | 2.08616e-6 | 3.30220e-9 |
| 96-ms island in five-second silence | 157 | 3.81842e-8 | 4.60471e-10 |
| Synthetic JFK self-overlap | 344 | 1.19209e-7 | 1.09376e-9 |
| MINDS Portuguese 0 | 283 | 1.31503e-6 | 7.96807e-9 |
| MINDS Portuguese 1 | 147 | 1.57021e-6 | 1.25570e-8 |
| MINDS French 0 | 118 | 4.47035e-8 | 2.69531e-9 |

All ten repetitions of every fixture pass both hysteresis decision gates, exact sample-span agreement from native/oracle probabilities, rounded-centisecond export comparison, fresh state, reset and final-operator cancellation/retry. Synthetic overlap is a shifted mixture of the same recording; natural overlap accuracy is untested.

Runs used an isolated network-disabled container, four-CPU quota, affinity CPUs 0–7, 8 GiB limit and zero process swap. No GPU was exposed. Live Qwen idle and host available memory of at least 6 GiB were checked before and during execution. The final trained runner exited zero without a guard abort.

## Limits

This qualifies the bounded native VAD graph and span policy against the selected reference. It does not establish acoustic accuracy, finite maximum-speech splitting, long-form quality, durable VAD resume identity or end-to-end Whisper performance. The opt-in Whisper integration currently rejects `word crosses a removed VAD gap; original timing unavailable` on JFK. No timing interpolation, tolerance increase or default-service change was used to bypass that failure.

Acceptance criteria: [native Go Whisper performance contract](../speech/whisper-go-performance-contract-20260930.md).
