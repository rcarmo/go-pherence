# Explicit Vulkan integer-dot groundwork — 1 October 2026

Intel packed integer-dot execution and an independent Q8_1/Q5 correction oracle pass. A standalone synthetic tiled projection is about 30% faster than the retained grouped-Q5/F32 operator, including activation quantisation. No Whisper model path or default changed. Original-engine arithmetic equivalence, trained performance and quality remain unqualified.

## Device and shader admission

`VulkanInitIntegerDot` explicitly requests Vulkan 1.3 `shaderIntegerDotProduct`, and checks accelerated packed mixed-signedness support. It refuses to recreate or upgrade an already initialised baseline device. Failure rolls back local instance/device/pool and optional symbols; the enabled flag is published only with successful native state. No Int8 or 8-bit storage is enabled, and no CPU/F32 fallback is supplied.

`VkKernelCreateIntegerDot` has its own admission envelope. The existing constructor/inspector still reject every optional integer-dot or quantisation fixture. The optional envelope requires the exact packed capabilities/extension, checks signed-left/unsigned-right int32 types and packed format on `OpSUDot`, and admits typed Round, FMax, half pack/unpack and signed conversions for Q8_1. Unknown operations and debug metadata remain rejected; these shaders are stripped and independently validated. This is metadata admission, not semantic or memory-safety verification.

ABI tests cover structure sizes/offsets, feature/property queries, chain enablement, missing symbols, unsupported/nonaccelerated devices, create/queue/pool failure rollback and refusal to upgrade baseline state. Shader-negative tests cover wrong packed format/types/extension, missing/duplicate declarations, other dot opcodes and disabled device. Vulkan-Headers/SPIR-V grammar pins and regeneration commands are recorded with [test fixtures](../../backends/vulkan/testdata/integer-dot/README.md).

## Native arithmetic checks

Physical device: Intel Iris Xe RPL-P, Mesa 26.1.5. The isolated runner used CPU4, memory8GiB with memory-plus-swap8GiB, read-only root, no network and the existing render node. Qwen idle and host available memory ≥6GiB guards remained active; serving state was unchanged.

- Packed unsigned weight bytes × signed activations: 1,025 sums × five repeats, exact int32 results.
- Q8_1: 205 blocks (1,845 packed words) × five repeats, bitwise scalar-oracle match. Covers zero, sign/extrema, nearest-even ties on this device, F16 subnormal scales, underflow and scaled sums.
- Q5_0×Q8_1: all 205 independent-block correction outputs × five repeats, bitwise scalar-oracle match. Original 22-byte weights are decoded independently, not via GPU packed words.
- Tiled matrix: all outputs on 1×32×1 and 33×96×63 cases match the independent block/FMA oracle. Four selected edge/interior outputs per full shape match; **full-shape scalar parity is not claimed**. Guard words, precancellation, reuse and native allocation cleanup pass.

GLSL `Round` permits implementation-defined tie direction. The oracle fixes nearest-even, and the native tie test passes on Sigma. Original shader uses eight lanes per block and a different reduction/division arrangement. This diagnostic uses a serial block maximum and reciprocal multiplication. Exact original-engine quantised-block parity has not been collected; the current check must not be called an original arithmetic oracle. Input nonfinite/overflow rejection and a public validated quantiser/operator API are not implemented.

## Synthetic projection timing

Same native process, four threads, five paired repeats. Retained grouped-Q5/F32 weights and the candidate use the same synthetic original-Q5 bytes and F32 input. Candidate time includes a separate activation-quantisation dispatch plus tiled integer-dot projection; both include native host submission/fence overhead. Preparation, model loading and request latency are not included. No samples were discarded from the five timed pairs.

| M×K×N | Grouped Q5/F32 median (ms) | Q8 quantise + integer-dot median (ms) | Change |
|---|---:|---:|---:|
| 1500×1280×1280 | 8.911 | 6.137 | −31.13% |
| 1500×1280×5120 | 32.895 | 22.462 | −31.71% |
| 1500×5120×1280 | 34.033 | 23.939 | −29.66% |

This is synthetic stage evidence, not pinned trained tensors or whole-model acceptance. Pair order is baseline then candidate, not randomised. Candidate uses F32 block correction and explicit FMA across blocks, rather than existing per-element ordered F32 FMAs. Maximum absolute drift from the F32 baseline was 0.105 for the square/FC1 shapes and 0.150 for FC2; mean absolute drift was 0.0201 and 0.0283. This drift is reported, not accepted via a widened tolerance. All arithmetic remains diagnostic and opt-in.

## Failures retained and verification

Initial admission used the wrong property-array index (8 instead of 5), so two native runs refused supported hardware. The independently collected `vulkaninfo` properties and complete acceleration array identified the error; the fixed ABI/index passes native execution. A test-only constructor argument swap caused small/non-square diagnostic failures; it was fixed before qualification. Q8_1 shader admission initially failed on unsupported syntax; source rewrites removed composite/equality instructions, leaving narrowly typed quantisation conversions as explicit additions. Host race invocation initially lacked cgo/compiler; the guarded build environment supplied them. These failures were not timing samples and are retained in evidence.

`make model-layout-check host-build host-vet host-test docs-check`, whole-tree race tests (`-p=2 -timeout=180s`), ARM64/RISC-V builds and the 28 retained production-shader regeneration/validation gate passed. An outer240s tool timeout did not drain its container: the remaining guard checks were maintained, actual container exit0/noOOM was verified before removal, and native work ran only afterward. Latest affected backend tests/vet passed after the pointer-pinning follow-up. Independent review timed out; no review approval is claimed.

[Hashed evidence](../../benchmarks/speech-foundations/vulkan-integer-dot-groundwork-20261001/) retains native logs, failed diagnostics, states, gate logs, parser totals, compiler provenance and runner. [Shader fixtures](../../backends/vulkan/testdata/integer-dot/README.md) retain exact source and stripped SPIR-V. No production service/default/resource changes or model integration occurred.

## Next gates

1. Compare quantised bytes/block scales with the pinned original quantiser on realistic layer inputs; resolve tie/reduction/overflow semantics explicitly.
2. Add a checked, separately named Q5×Q8_1 operator/set and dedicated activation scratch in the original-Q5 FFN candidate. Existing F32 path stays unchanged; reject invalid inputs/device state.
3. Qualify trained tensor outputs and encoder drift before five-repeat EN/PT/FR and native VAD/word/silence trials. Treat quantised activation accuracy as a new gate.
4. Compare matched preparation/request/full-workflow boundaries against original; retain independent timing, long-form/resume/quality and fault-coverage requirements.

The original-speed goal remains active and unmet.
