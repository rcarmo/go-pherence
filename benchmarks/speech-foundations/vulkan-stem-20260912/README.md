# Vulkan convolution and residual primitives

`VkConv1D3F32` and `VkAddF32` passed static, mocked and native Iris Xe tests. A resident two-convolution stem with GELU and position addition matched five separately fenced GPU dispatches bit-for-bit. The full Whisper encoder and trained checkpoints have not been wired to these operators.

## Contracts

`VkConv1D3F32` implements kernel size 3, symmetric zero padding 1 and stride 1 or 2, without groups or dilation. Inputs may be channel-first `[inCh,inLen]` or time-major `[inLen,inCh]`; weights are `[outCh,inCh,3]`, bias `[outCh]`, and output is always time-major `[ceil(inLen/stride),outCh]`. Input length is 1–4096; input/output channels are 1–2048. Bias is required; use a zero tensor when absent.

The shader uses 16×16 shared input/weight tiles (2048 bytes) and computes patches while loading each tile. No full im2col buffer, host transpose or activation fusion is involved. Tap indices stay unsigned: the left padding check precedes subtraction of one, and the right-padding check precedes input access. Tail lanes load zero and reach both barriers. Each output element has one writer. `OpISub` is the only addition to the existing bounded opcode envelope, with arity/result metadata checked against the previously pinned Khronos core grammar.

`VkAddF32` uses the unchanged vector-add shader: matching contiguous ranks 1–8, three descriptors, four push bytes, local size 256. It supports exact output alias with either or both inputs, but rejects partial output overlap. There is no broadcasting. Convolution rejects every output/input overlap; read-only inputs may alias.

Both wrappers validate exact shapes and byte extents, checked dimension/count bounds, device grids/ranges and lifetime ownership before recording. Caller values must have finite representable products/sums; no content scan occurs. `Stage` returns the existing mutable generic stage type. Plans revalidate generic resources only, not convolution/addition invariants; callers modifying a stage assume shader safety responsibility. The source review highlighted this distinction, and the new API comments state it explicitly.

## Static and offline checks

- Ten new top-level tests; full mock selection: 125 top-level tests, 434 passing test/subtest events, no failures/skips.
- 58 operator/plan/shader tests ×30 shuffled repeats: 5520 passing events, counted separately.
- Convolution source model: 11 shapes ×2 layouts ×4 lane/group schedules, 46,248 values. Independent direct float64 convolution oracle; budget `2e-5 + 2e-5*abs(reference)`. Maximum absolute error `2.0864453687430284e-5`, within the combined absolute/relative budget.
- Layout pairs and reordered schedules are bit-identical. Asymmetric analytic taps test kernel orientation, boundary padding and bias. Output ownership/canaries, maximum metadata-only shapes, nil/closed ownership, stride/layout rejection, all output aliases, push/range/grid capture and copied stage slices are covered.
- Addition tests cover exact alias with either/both inputs, copied views, disjoint output, mixed exact/partial aliases, ranks 1–8, uint32 count overflow and descriptor/grid limits.
- Convolution→in-place add plan cancellation retains both kernels and the arena until confirmed drain; constructor and idempotent close mocks pass.
- Sixteen embedded and sixteen rebuilt shaders pass `spirv-val --target-env vulkan1.3`. Six modules rebuild byte-identically, including convolution; all 16 narrowly normalised comparisons pass. Checker: six Bun tests, 55 assertions.

The source review found no scoped shader padding/index/orientation/barrier issue. Its stage-mutability concern was addressed by precise API documentation, not an immutable-stage redesign. No claim is made that generic plans enforce operator-specific invariants after caller mutation.

## Native Iris Xe checks

Device/driver selection and the opt-in harness are the same as the [first native checkpoint](../vulkan-native-20260912/README.md). No timing mode was enabled for this checkpoint. Three final repeats passed all 39 test/subtest events, with zero failures/skips. The complete suite contains 339 numerical cases and 610,056 compared values, including regression fixtures for earlier operators.

New component results, with repeats counted separately:

| Component | Cases | Values | Maximum absolute error |
|---|---:|---:|---:|
| Convolution | 66 | 34,686 | 2.0864453687430284e-5 |
| Addition | 90 | 19,275 | 0 |
| Two-convolution stem | 27 | 56,691 | 1.1920928955078125e-7 |

Convolution uses the same predeclared `2e-5 + 2e-5*abs(reference)` budget; addition requires exact F32 output; the composed stem uses `5e-5 + 5e-5*abs(reference)`. No tolerance changed after a native result. Native CF/TM convolution outputs match bit-for-bit.

Stem shapes are `(input length, mel channels, output channels) = (17,2,3), (65,80,64), (129,128,64)`. The sequence is CF convolution stride1 → in-place erf-form GELU → TM convolution stride2 → in-place GELU → in-place position addition. These include Whisper's 80/128 mel input widths but reduced output channels; no full trained-size stem ran. Input and weights stay resident, with no intermediate host transfer. Each final plan is bit-exact against five separately fenced GPU dispatches; all nine repeat/shape comparisons pass.

Guard tensors and return-to-baseline live-allocation checks pass. Initial native tests and a separate make-target run also pass and are kept separately from the three-repeat totals. No native cancellation/device-loss recovery was attempted.

## Resource state and reproduction

Rui's compute authorisation remains in effect. The LLM and both speech services stayed inactive/dead with MainPID zero. Before/after `pswpin=36292` and `pswpout=635627` are unchanged. These are bounded synthetic correctness runs, not performance measurements or continuous resource telemetry.

```sh
make speech-vulkan-static-check VULKAN_SHADER_REPORT=/tmp/new-stem-static
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-offline-check speech-affine-check speech-foundations-check speech-media-integration

# In an authorised window, selecting the installed GPU ICD explicitly:
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-native-check
```

[Evidence](evidence.json), [static report](static/verification.json), [disassembly](conv.spvasm) and raw logs are included. Focused make regressions, Vulkan/board vet, affected builds and arm64 test cross-build pass. Full-tree/backend compile failures match the prior checkpoint; the race build still lacks `gcc`.

No existing shader defaults, model inference path, service data or media backend changed. Quantised formats, resident encoder wiring, full-model quality/performance and device recovery are unfinished. Strict SincNet retains four failures. Nothing was pushed, deployed or restarted.
