# Exact Vulkan decoder GEMV feasibility — 1 October 2026

Standalone Vulkan matrix-vector projections lose to the accepted four-worker CPU path at decoder Q/K/V/O and FFN shapes. Only the vocabulary-head shape shows a small transfer-inclusive gain in the final synthetic run. Pinned head inputs preserve all logit bits, but no whole-request gain is measured. Per-projection offload and a head-only opt-in are declined; the retained decoder and defaults are unchanged.

## CPU reduction reproduced on Vulkan

The diagnostic F32 shader matches the existing amd64 `Sdot` reduction for K divisible by16. Eight lanes maintain the two independent FMA chains used by the CPU's two8-wide accumulators. Each lane adds its chains, then the output lane performs the same low/high128-bit and horizontal pair order. The final optional bias addition is unchanged.

Three geometries use local8×8,8×16 and8×32 workgroups: one eight-lane row per output, with shared reduction arrays of256/512/1024 bytes. Original finite stored weights and activations stay F32; no quantisation, new optional features, SIMD rewrite or tolerance changes occur. K tails are not admitted by this shader; output-row tails are masked.

The Vulkan closed contract and offline `spirv-val` admit all three shaders without expansion. Raw plans bind four buffers and8 push bytes. The vocabulary diagnostic uses51,866 outputs directly through the checked plan/buffer interface; it does not widen the normal `VkLinearF32` constructor's16,384-dimension limit or create a runtime model backend.

## Dispatch and transfer timings

Each configuration checks K16/N7, K32/N519, square K1280/N1280, FC1 K1280/N5120, FC2 K5120/N1280 and head K1280/N51866. Every synthetic output bit matches the four-worker CPU `Sdot`+bias result. Guards and normal native cleanup pass on all18 geometry/shape cases.

The CPU reference partitions output rows like the accepted decoder option. The resident GPU arm includes plan submission and fence completion, with input/output already resident; the transfer arm also uploads the input and downloads the result. Weights remain resident in both GPU arms. Five samples each contain ten calls, including the first timed sample after reference construction. The initial run has fixed CPU→resident→transfer order; the final run reverses that order on odd repeats. Both raw runs are retained. There is no clock/temperature control or discarded sample.

Final8×32 medians:

| K/N | CPU4 (ms/call) | GPU resident (ms/call) | GPU transfer (ms/call) |
|---|---:|---:|---:|
|16/7 |0.005 |0.160 |0.142 |
|32/519 |0.028 |0.183 |0.172 |
|1280/1280 |0.180 |0.258 |0.244 |
|1280/5120 |0.689 |0.895 |0.874 |
|5120/1280 |0.635 |1.102 |1.188 |
|1280/51866 |6.171 |5.909 |5.919 |

The smaller projections do not recover submission/fence costs. Initial8×32 head timing was6.596/5.021/5.325ms, a larger isolated advantage than the final result. Absolute values vary between processes; final head transfer advantage is about4.1%. The measured conclusion is limited to these kernels/shapes and this Intel device. A fused resident decoder could amortise dispatch differently and was not implemented or tested.

Resident versus transfer timing is a workflow boundary, not a guarantee that transfer always increases the median: scheduling/noise can reverse close measurements. No per-arm additive decomposition or whole-decoder speed prediction uses those differences.

## Pinned vocabulary projection

The model diagnostic loads the SHA256-verified original turbo Q5 source and checked HF metadata, widens the original token-embedding matrix to the same F32 values used by the CPU decoder, and copies that matrix into a Vulkan arena. Its payload is265,553,920 bytes (about253.25MiB), additional to the existing CPU weight storage. Upload time is recorded separately; full-model peak RSS or request setup is not measured.

A CPU decoder with deterministic1500×1280 encoded input executes eight fixed tokens. After each token, the final layer-normalised1280-value input is captured. The GPU head compares all51,866 output logits against the CPU head result. Five alternating-order CPU/GPU calls per input check every bit. The GPU wall includes input upload, plan execution, output allocation and download; the CPU wall includes its normal four-worker output allocation/projection.

| Captured decoder step | CPU median (ms) | GPU transfer median (ms) | Change |
|---|---:|---:|---:|
|0 |5.945 |5.698 |−4.16% |
|1 |5.907 |5.730 |−3.01% |
|2 |6.553 |5.755 |−12.18% |
|3 |6.015 |5.676 |−5.63% |
|4 |5.928 |5.723 |−3.47% |
|5 |5.897 |5.747 |−2.55% |
|6 |5.959 |5.727 |−3.90% |
|7 |5.968 |5.712 |−4.30% |

All logits match by F32 bits. Pre-cancelled plan execution returns cancellation; subsequent fresh execution, guards and final allocation/pipeline cleanup pass. These are captured inputs from the CPU decoder, not a GPU decoder token trajectory. The synthetic encoded input is not speech-derived. No expanded acoustic/word-timing, complete hidden-state, in-flight device-fault or long-form/resume acceptance follows.

The head result could benefit another workload. For this path, a second resident copy and lifecycle seam would buy only a small fraction of the already-small vocabulary-head contribution. A head-only integration is declined without extrapolating a request gain. Full decoder residency/fusion remains untested.

## Retained source and checks

All diagnostic shaders and Go tests are removed from runtime/test discovery. The accepted CPU row-scheduling option, GPU encoder, SIMD helpers and generation/alignment logic remain unchanged. The31-shader runtime inventory is unchanged.

Build corrections were limited to demonstrated API mismatches: `NewVkTensorArena` takes an `int` capacity, and `AllocF32` requires a context before dimensions. Those compile failures occurred before native execution. Corrected synthetic and pinned native diagnostics pass; diagnostic vet passes.

The restored tree passes `make model-layout-check host-build host-vet host-test docs-check`, affected Vulkan/Whisper race tests with accepted decoder row scheduling enabled, and marked ARM64/RISC-V builds. Final documentation checks follow this report.

A read-only feasibility judge completed on supplied timing/test facts. It confirms that the results do not establish general decoder speed, request improvement or acoustic acceptance. It did not inspect raw source/evidence; no independent source audit is recorded.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-decoder-vulkan-gemv-20261001/) contains all three shaders/SPIR-V, generator, synthetic and pinned-head harnesses, initial/final raw timing logs, input/model settings, parser/comparison, binary/tool provenance, guards, container states/exits and retained gates. No weights, audio or executable binaries are committed.

The fresh @llama hold excludes competing experiments/builds/restarts. CPU4/8GiB/no-swap, physical Intel GPU, Qwen-idle and host available memory≥6GiB guards remain active. Pinned-head inference uses heap4GiB. Native deadlines remain≤120s. Successful containers exit0/noOOM and actually drain before explicit release. Defaults/services/resource allocations, Qwen LAN and Gemma are unchanged.

The original-speed, VAD `Thank you.` disagreement, independent acoustic/word timing and long-form/fault goals remain open. These standalone tests rule out this per-projection approach on the tested shapes; they do not rule out a resident fused decoder or a separately quality-qualified original-compatible arithmetic path.
