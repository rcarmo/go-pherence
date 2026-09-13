# Vulkan Q8 linear weights — 13 September 2026

The standalone per-row Q8 weight operator reduces the tested projection storage by about 75% and passes native Iris Xe kernel-parity gates. It is slightly slower than the F32 operator and has measurable weight-quantisation error, so no model graph or default selects it.

## Format

`NewVkLinearQ8WeightF32` quantises each complete output row with:

- `scale[row] = max(abs(weight[row])) / 127`;
- nearest-even integer rounding through Go `math.Round` for non-halfway generated inputs;
- signed values clamped to `[-127,127]`;
- four little-byte-first signed values per `uint32`.

Zero rows use scale zero and packed zero values. NaN/Inf fail. Packed bytes and an alignment-safe F32 scale range share one owned host-visible/coherent Vulkan allocation. X, bias, accumulation and output remain F32.

The shader extracts each byte with 32-bit shifts/masks, converts through core 32-bit arithmetic, applies the output-row scale and uses the same 16×16 F32 reduction schedule as the baseline linear operator. SPIR-V declares only `OpCapability Shader`, `OpTypeInt 32` and `OpTypeFloat 32`. It requests no `Int8`, `StorageBuffer8BitAccess` or integer-dot capability.

This is weight-only Q8. It is not GGML Q8_0/Q5, does not quantise activations and cannot enter `VkF32Plan`.

## Verification

- Exact scale, signed-byte packing, padding and dequantisation tests cover odd and even shapes.
- Source-model tests use the exact dequantised weights and checked tile schedule.
- Descriptor tests verify aligned packed/scale ranges, push words, groups, aliases, pending retention, drain and copied-owner invalidation.
- Twenty shuffled focused repetitions pass.
- All 23 embedded/stored/rebuilt shaders pass offline validation; the Q8 shader rebuild is byte-identical.
- GLSL SHA-256: `05035ef711ce3d1329a0ad329017fb7aed9d7e5f5fe8149a9c8c488c897dcae1`.
- SPIR-V SHA-256: `b0154a895beabd1bba03131ddaeb5ff3cde99a356bb6dffe26688fb0db47bae4`.
- Disassembly SHA-256: `27f4149bdef673b925fd4cf5a35225307b2d42360cd9d2be330f6d94bff0549e`.
- Static verification JSON SHA-256: `bd3bba62a3323918a6f7566b94094c3d3e8be27e39b39589fdaadb748f97e21e`.

## Native result

Device: `Intel(R) Iris(R) Xe Graphics (RPL-P)`. Seven shapes cover odd dimensions and `K=1,3,16,31,65,384,1280`.

Kernel output is compared with a CPU oracle using the exact dequantised Q8 weights. Maximum absolute kernel error is `4.0808872756037395e-7`, below `2e-5 + 2e-5*abs(reference)`. Separate comparison with the original F32 weights records lossy quantisation: RMS error/reference-RMS ratios rise from `0.056%` at `K=3` to `0.656%` at `K=1280`; the largest sampled absolute error is `0.0032397192532477026`.

One full encoder projection used identical inputs, bias and dequantised weights for both timing paths:

| Shape | F32 weight bytes | Q8 storage bytes | F32 median | Q8 median | Speedup |
|---|---:|---:|---:|---:|---:|
| `1500×1280×1280` | 6,553,600 | 1,643,520 | 28.585ms | 29.035ms | 0.984× |

The Q8 byte count includes aligned row scales. Six host-wall dispatch samples per operator were collected in three alternating blocks. Construction, quantisation, upload and download are outside the samples. GPU timestamps are unavailable. The process took `0.80s`, used `95,180KiB` maximum RSS and reported zero swaps; host swap counters did not change.

Test log SHA-256: `e3c55a6c70fa97c3d3aa7578147425ef7f59a15a2341867e3eeb48dad25bdd08`. Time output SHA-256: `b75dbc923662b1974fce2bea00e957f365d3ad3d7c231b88532ba5f783a73ade`.

## Decision

The format and numerical implementation pass their scoped gates. The measured operator is slower than F32, and no model-level WER/DER tolerance exists for this Q8 representation. It remains an explicit diagnostic candidate. Future quantised work needs an integer activation/dot-product path or another measured format, plus complete model-quality qualification.

## Reproduce

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
GO_PHERENCE_TEST_VULKAN_LINEAR_Q8_TIMING=1 \
  GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-linear-q8-weight-check
```
