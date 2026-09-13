# Vulkan packed-F16 linear weights — 13 September 2026

The new standalone linear operator halves resident weight storage and passes native Iris Xe numerical gates. It is slightly slower than the existing F32 operator for the tested full encoder projection, so no model graph or default selects it.

## Format and arithmetic

`NewVkLinearF16WeightF32` accepts finite row-major F32 weights `[N,K]`, rounds them to IEEE-754 F16 and owns two little-half-first values per `uint32` storage word. Finite source values that round to F16 infinity fail. Odd tensors zero the unused upper half.

The shader accepts F32 `X[M,K]`, F32 bias and F32 output. It widens packed weights with core 32-bit integer/float operations before F32 multiplication and accumulation. Its SPIR-V declares only `OpCapability Shader`, `OpTypeInt 32` and `OpTypeFloat 32`; it does not request `Float16`, 16-bit storage or another optional device feature. This is reduced weight storage and bandwidth, not native F16 arithmetic.

The operator owns one immutable dedicated weight allocation and one kernel. Composite close preflights both resources before freeing either one, retains them while a submission is unresolved, and closes in reverse construction order. The existing `VkF32Plan` accepts only `*VkTensorF32`, so this mixed-storage operator cannot be inserted into an F32 plan. No dtype checks were weakened.

## Offline gates

- Exact host packing/rounding and odd-tail checks cover 1–257 values.
- NaN, Inf, F16 overflow, malformed geometry, byte extents and output aliases fail before dispatch.
- Source-model tests cover six projection geometries through `K=1280` against exact rounded-weight float64 oracles.
- Mock dispatch checks all descriptor ranges, push words, groups, cancellation retention, drain and idempotent close.
- Twenty shuffled focused repetitions pass.
- All 22 embedded/stored/rebuilt shaders pass the existing offline validator/compiler check.
- New GLSL SHA-256: `931a83b0bda03464885dfffe5fe786d2581ee11a4867e59160c4303d9a7c625c`.
- New SPIR-V SHA-256: `7fc207d2b71cf9aac0f77308159382cb509ed27dbc187cac76f18ff9293945d4`; the rebuild is byte-identical.
- Disassembly SHA-256: `000ff757873635537ee5f2ad596704e50e9e011faf643559cebe661a55ea8077`.
- Offline verification JSON SHA-256: `9d0561aa0ef6504b79e0919b92c6188f68df44117ae5c43a267e47787c76e3f1`.

## Native result

Device: `Intel(R) Iris(R) Xe Graphics (RPL-P)`. Seven shapes cover odd dimensions and `K=1,3,16,31,65,384,1280`. Maximum absolute error against the exact rounded-F16-weight oracle is `3.029892964079295e-6`, below `2e-5 + 2e-5*abs(reference)`. Allocation/state returns to its initial snapshot.

One full encoder projection used identical inputs, bias and rounded weights for both operators:

| Shape | F32 weight bytes | Packed weight bytes | F32 median | Packed median | Speedup |
|---|---:|---:|---:|---:|---:|
| `1500×1280×1280` | 6,553,600 | 3,276,800 | 28.671ms | 29.281ms | 0.979× |

The measurements use six host-wall dispatch samples per operator in three alternating blocks. Construction, packing, upload and download are outside the samples. GPU timestamps are unavailable. Maximum packed-versus-F32 output difference is zero because both paths use the same rounded weights and reduction schedule.

The native test process took `0.80s`, used `95,312KiB` maximum RSS and reported zero swaps. Host `pswpin`/`pswpout` counters did not change. Test log SHA-256: `d69956a4ee4c7d888d8ae87070a8463bdefdfff8649634283d6da805c3137bcd`. Time output SHA-256: `b47db33556ff02a23ad319f12c3307b1693e62fc0f879be10e5b1b2e8609e408`.

## Limits

The candidate reduces weight bytes but did not improve the tested projection latency. It has no F16 activation/output, attention, convolution, recurrent or model integration. It does not load GGML Q5/Q8 formats and makes no quantised quality claim. Model-level F16/quantised work remains open.

## Reproduce

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
GO_PHERENCE_TEST_VULKAN_LINEAR_F16_TIMING=1 \
  GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-linear-f16-weight-check
```
