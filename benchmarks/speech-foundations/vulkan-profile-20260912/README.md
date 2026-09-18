# Turbo encoder operation profile

Linear projections dominate the current resident F32 Turbo encoder: about 77% of the separately fenced diagnostic time, followed by attention at about 20%. The normal encoder still takes about 14.23 seconds. No faster kernel was implemented or selected in this checkpoint.

## Same-work measurement

The profile uses the pinned trained Turbo weights and the canonical public Portuguese row-0 PCM from the [Turbo baseline](../vulkan-turbo-20260912/README.md). The exact frontend produces the normal padded 3000-frame input. Weights and activations stay resident. The public encoder constructor and runtime defaults are unchanged.

A private constructor hook captures the actual operator stages used by the 34 normal plans. It then creates 390 single-stage plans over the same kernels/tensors. Both execution modes run the identical mathematical graph. Before each mode the same mel tensor is uploaded; downloads and final-output checking occur after the timed region. Each final tensor must match the normal `Forward` reference bit-for-bit.

Each of two final processes runs a reference forward plus two alternating rounds:

1. normal layer plans → separately fenced stages;
2. separately fenced stages → normal layer plans.

This yields four normal-mode and four diagnostic-mode measurements. Each diagnostic pass has one submission/fence per operator instead of per layer. Timings include command recording, descriptor setup, submission and wait overhead. They are not GPU timestamps or isolated shader duration. The additional 356 submissions can inflate individual categories; these measurements guide kernel investigation, not a claim of hardware traffic attribution.

## Results

| Final process / round | 34 normal plans | 390 single-stage plans | Final output |
|---|---:|---:|---|
| 1 / 0 | 14.234 s | 14.570 s | Bit-exact |
| 1 / 1 | 14.225 s | 14.572 s | Bit-exact |
| 2 / 0 | 14.231 s | 14.534 s | Bit-exact |
| 2 / 1 | 14.227 s | 14.536 s | Bit-exact |

All outputs have SHA256 `c523d40468f88d2acdea8db84c52d31e24b9d72c8c177a957dee4b5b709aede6`. This is encoder-output reproducibility on one real-speech input, not an additional WER corpus result.

Approximate operation totals per separately fenced pass:

| Category | Calls | Time per pass |
|---|---:|---:|
| Linear 1280→1280 (Q/K/V/output) | 128 | 3.72–3.79 s |
| Linear 1280→5120 (FC1) | 32 | 3.75–3.78 s |
| Linear 5120→1280 (FC2) | 32 | 3.74–3.77 s |
| Attention | 32 | 2.884–2.886 s |
| All convolutions, norms, activations and additions | 166 | about 0.37 s |

The three projection families perform equal aggregate multiply/add work under the shape-based FLOP estimate. The attention estimate includes QK and weighted-value products, but excludes softmax and other overhead. Effective arithmetic rates include dispatch overhead and must not be described as peak GPU rates.

The extra submissions add roughly 0.30–0.35 seconds overall in final runs, while the ordinary encoder remains near 14.23 seconds. The profile points first to linear-kernel efficiency; it does not determine whether F16, quantisation, a different tile, packing or another scheduling change is the right solution. Every candidate still needs numerical, transcript and whole-request checks. Attention is the next measured category.

## Ownership and failure checks

`NewVulkanEncoder` still calls the same `vk.NewVkF32Plan`. Its plan factory was extracted into private `newVulkanEncoder` for fault injection and diagnostic capture. Private factories returning a nonnil plan with an error are now adopted before rollback; nil successful results are rejected. No generic stages or profiling callbacks are exposed through public inference APIs.

Review found the partial-factory cleanup gap and highlighted the different submit counts and opaque driver allocations. The cleanup gap was fixed. The measurement and memory limitations are explicit here. Native fault injection verifies that a returned partial plan is closed, nil-success construction fails, and live native allocations return to baseline.

Four offline encoder tests pass 30 shuffled repeats: 600 test/subtest events. Three native reduced-encoder suite repeats pass 30 events, including three partial-plan rollback checks, prior ownership/cancellation checks and PCM bridge tests. Two final profiling processes pass with all eight mode outputs bit-exact. An initial profiling run passed separately and is retained as preliminary evidence.

Focused speech/Vulkan/affine/media regressions, affected vet and amd64 builds pass. Full-tree/backend and Whisper arm64 FFT errors match the prior baseline; race compilation still lacks `gcc`. The new make target was dry-run checked; the equivalent native test command was executed for the profiles.

## Resource scope

The profiler allocates the same 2,641,735,680 explicit native bytes across 34 arenas as the baseline. It also creates 390 descriptor pools/sets, command buffers and fences; their opaque driver allocations are not counted in `VulkanMemoryStats` or the 4-GiB/40-allocation cap. Construction is outside timed regions. The host model and weight-bearing layout descriptions are released after upload; the symbolic operation list retains no host weight slices.

One-second monitoring across 158 final samples recorded minimum MemAvailable 21,517,024 KiB (about 20.52 GiB). Swap counters stayed `pswpin=36297`, `pswpout=635627`. The LLM and both speech services stayed inactive/MainPID zero. These samples do not prove absence of sub-second memory/thermal pressure. No GPU counter, energy or allocation/GC study was performed.

After profiling, the go-264 owner received a separate bounded CPU-only Tiny admission window; no paired media-model run overlapped these GPU profiles. The optional adapter remains outside the active checkout, awaiting reviewed licensing/version handoff. FFmpeg stays the default.

## Reproduction

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
export GO_PHERENCE_WHISPER_TURBO_DIR=/path/to/verified/whisper-turbo-41f01f3
export GO_PHERENCE_MINDS_FIXTURE_DIR=/path/to/verified/MINDS-cache
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-turbo-profile
```

Run only in an authorised compute window. The test requires a ≤300-second process deadline, 270-second context, pinned weights/audio and initially empty native allocation accounting. There are no downloads, model defaults or service lifecycle actions.

[Evidence](evidence.json), [operation aggregates](metrics.json), raw per-stage samples and resource snapshots are included. Performance is still on hold; no speedup or precision-format change is claimed. Quantised/native-F16 execution, broader model quality, Community-1, device recovery and whole-job targets remain unfinished. Nothing was pushed, deployed or restarted.
