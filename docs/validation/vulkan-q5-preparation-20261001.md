# Exact Q5 preparation trials — 1 October 2026

None of the prepared-weight variants is retained. Signed-byte preparation slows trained FFN kernels by 11–14%. A paired offset-byte layout gains about 2% at the operator level, but increases packed storage by 50% and substantially increases CPU packing cost. Widening only scales keeps storage unchanged but slows trained kernels by 5.8%. No end-to-end gain, model admission or default change is claimed.

## Arithmetic and storage

The experiment moves Q5_0 bit reconstruction from repeated shader work into one-time CPU preparation. It does not quantise activations or change the increasing-K F32 projection accumulation. Original signed weights remain integers in −16..15; the original finite F16 scale widens exactly to F32. The shader still multiplies each weight by its block scale before the unchanged ordered projection arithmetic.

| Representation | Bytes per 32 values | GPU reconstruction |
|---|---:|---|
| Original file | 22 | Original F16 scale/high bits/nibbles |
| Retained decode4 storage | 24 | Original bits; scale padded to uint32 |
| Prepared signed bytes | 36 | F32 scale + 32 two's-complement bytes |
| Prepared offset bytes | 36 | F32 scale + 32 values coded as 0..31 |
| Scale-only widening | 24 | F32 scale + unchanged high bits/nibbles |

The first signed shader emitted `OpShiftRightArithmetic` and signed-to-float conversion outside baseline admission. Native construction failed with `unsupported or malformed Vulkan shader contract: opcode/word framing`; no dispatch occurred. The corrected shader extracts an unsigned byte and computes `float(byte & 127) - float(byte & 128)`. These integer-valued F32 operations are exact. Admission remains closed and unchanged. Offset variants use `float(byte) - 16.0`.

The contiguous layouts assign each lane eight adjacent K values; `pair4` instead retains the decode4 lane assignment of four low-index and four high-index values, with pairs interleaved in the prepared byte stream. Both use two packed words per lane. Only the layout differs; accumulation order does not.

## Synthetic five-repeat diagnostic

Intel physical GPU, 1500 rows, FFN shapes K1280/N5120 and K5120/N1280. Five alternating-order baseline/candidate samples include the first. All complete outputs match the retained decode4 reference bit-for-bit; a 33-row/K96/N63 tail case and both allocation guards also pass.

| Variant | FC1 time change | FC2 time change | Decision |
|---|---:|---:|---|
| Signed bytes, 4 lanes/block | +11.34% | +12.07% | Reject |
| Signed bytes, 8 lanes/block | +27.41% | +27.18% | Reject |
| Signed bytes, 16 lanes/block | +30.87% | +30.42% | Reject |
| Contiguous offset bytes, 4 lanes/block | +17.90% | +16.13% | Reject |
| Paired offset bytes, 4 lanes/block | −2.90% | −1.60% | Trained follow-up only |
| F32 scale with original packed Q5 | +6.10% | +6.42% | Reject |

These figures come from the initial successful processes. The final diagnostic reruns the five byte variants on the retained harness; scale-only has its own successful synthetic/trained process. No variant's speed is added to historical model gains.

## Pinned trained projection measurements

Pinned original turbo Q5_0 file SHA256 `394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2`. Block-0 FC1/FC2 tensors and previously retained, independently SHA256-pinned `ff-norm`/`tmp-ff` F32 inputs are used. This is an operator experiment, not a new encoder or speech request.

Final five-repeat trained byte-layout results and the separate five-repeat scale-only results:

| Variant | Tensor | Decode4 median (ms) | Prepared median (ms) | Change |
|---|---|---:|---:|---:|
| Signed bytes | FC1 | 31.731 | 36.152 | +13.93% |
| Signed bytes | FC2 | 32.792 | 36.529 | +11.39% |
| Paired offset bytes | FC1 | 31.719 | 31.149 | −1.80% |
| Paired offset bytes | FC2 | 33.461 | 32.800 | −1.97% |
| F32 scale only | FC1 | 30.796 | 32.570 | +5.76% |
| F32 scale only | FC2 | 31.734 | 33.563 | +5.76% |

Initial trained byte-layout measurements also found signed slower (+10.93/+11.66%) and paired offsets slightly faster (−2.08/−1.75%). There is no trained timing win for signed preparation; the paired-offset result is too small to justify integrating the costlier representation. No full encoder, expanded speech/VAD/word or hidden-encoder gate was attempted for these rejected variants.

Each comparison checks all 7,680,000 FC1 and 1,920,000 FC2 output values against the original decode4 operator, rather than sampling. Pre-cancelled execution returns cancellation; subsequent fresh execution matches every reference output bit. Native allocation/pipeline cleanup and guards pass. This is not an in-flight cancellation/fault qualification or an independent scalar projection oracle.

## Preparation and memory costs

Each trained matrix occupies 4,915,200 bytes in retained packed storage, versus 7,372,800 bytes in the byte layouts: +2,457,600 bytes, or 50%. Scale-only remains 4,915,200 bytes. The harness records logical storage and `runtime.MemStats.TotalAlloc` deltas separately from kernel time; these are not resident-memory measurements.

Final per-matrix CPU packing medians:

| Variant | FC1 retained → prepared (ms) | FC2 retained → prepared (ms) |
|---|---:|---:|
| Signed bytes | 1.368 → 17.562 | 2.058 → 18.029 |
| Paired offset bytes | 1.765 → 21.271 | 0.983 → 21.847 |
| Scale-only, separate process | 0.989 → 2.586 | 1.019 → 3.744 |

Packing currently creates a host temporary. Scale-only creates the original packed temporary and then the widened-scale result, so its setup figures do not estimate an optimised direct-write packer. The operator slowdown already rejects it; no direct-write implementation is needed.

For all 64 turbo FFN matrices, byte layouts would increase logical packed storage from 314,572,800 to 471,859,200 bytes (+150 MiB). This is a shape-based extrapolation, not an allocated full-model result or RSS claim. No model mode was created, so request preparation cost, full-model peak RSS and cold/warm request speed were not measured. The tiny paired-offset kernel gain cannot establish a request gain.

## Checks, corrections and retained tree

The offline harness checks all 32 signed Q5 values, positive/negative zero, subnormal and normal scales, and every finite F16 scale bit pattern. Nonfinite scales, malformed extents, nil contexts, unknown variants and bounded packing cancellation are rejected. Native admission and offline `spirv-val` pass for all corrected shaders; no contract extension, tolerance change, activation requantisation or CPU fallback occurs.

An added scale-only offline oracle initially indexed the nine-word layout on a six-word block and panicked before native execution. The oracle was corrected to branch on representation; the final offline/native harness and diagnostic race/vet checks pass. The read-only review delegate timed out after 60 seconds; no independent review approval is claimed.

The diagnostic Go test and all prepared shaders are removed from runtime/test discovery. Their exact source, SPIR-V, generator, parser, input/settings pins, raw logs, state/exit/monitor records and harness are archived under [hashed evidence](../../benchmarks/speech-foundations/vulkan-q5-preparation-20261001/). Retained source remains the previously accepted outputILP/decode4 implementation. The unchanged tree passes `make model-layout-check host-build host-vet host-test docs-check`; final affected race/cross/docs gates are recorded alongside the experiment.

Runs used the existing CPU4/8GiB/no-swap, physical Intel GPU, Qwen-idle and host-available-memory ≥6GiB guards. Trained processes additionally set `GOMEMLIMIT=4GiB`. Native deadlines remained ≤120s. Every successful container exited 0 without OOM; the first admission-failure container exited nonzero and drained. Defaults, services, resource allocations, Qwen LAN and Gemma remain unchanged.

The original Whisper Vulkan+flash+VAD speed/quality goal remains open. These rejected trials narrow the next work to scheduling/dataflow improvements that keep the existing compact representation, or independently quality-qualified arithmetic—not presenting small isolated gains as model acceptance.
