# Wide-load MMQ projections — 2 October 2026

The explicit Q5_0×Q8_1 MMQ projection kernel now reads Q8_1 activations as `uvec4`. Per-output arithmetic is unchanged, and native outputs are bit-identical. On the private `vulkan-original-q5-padded-integer-dot-mmq-tanh` backend, with GPU cross K/V and the fast Q5 decoder:

- the JFK encoder drops from 3.49 s to 3.20 s;
- the JFK request drops from 3.825 s to **3.506 s**.

All 10 fixtures produce identical output. Pinned whisper.cpp remains faster on JFK (about 3.02 s). Defaults, services and resources are unchanged.

## Attribution

A test-only harness timed 16 identical dispatches per plan, which removes per-submit fence cost, for the base kernel and source variants with parts removed. Variants with removed parts produce wrong values and are timing-only. Times are per dispatch in milliseconds; quantisation is timed separately. Evidence and sources are under `attribution/`.

| Shape | Base | No B (activation) loads | No A (weight) loads | Neither | No compute | **uvec4 B** | uvec4 B + uvec2 A |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1500×1280→1280 | 3.41 | 2.71 | 3.19 | 2.53 | 2.39 | **2.82** | 2.86 |
| 1500×1280→5120 | 12.35 | 10.66 | 11.82 | 9.86 | 8.99 | **11.07** | 10.95 |
| 1500×5120→1280 | 15.96 | 10.40 | 14.92 | 9.63 | 11.03 | **11.33** | 11.33 |

The scalar loads of B (activations) were the largest removable cost, especially for FC2 (K=5120). Wide B loads recover most of it with bit-identical outputs. Wide A loads add nothing: the driver already merges the adjacent weight-word loads. The original's GPU-timed matmuls, including quantisation, take 2.99 / 9.76 / 10.36 ms. QKVO is now at parity; FC1 and FC2 remain about 10–15% slower.

## Changes

- `shaders/integer-dot/linear-mmq.glsl` views the activation binding as `uvec4`. Each row's four-block step is 36 contiguous words, starting on a 16-byte boundary because K%128==0. The shared-memory layout, integer sums, `precise` epilogue, block-ordered FMA and bias are unchanged. Fixture and hashes are updated, and the integer-dot regeneration gate passes.
- SPIR-V admission: explicit integer-dot mode admits storage arrays of `uvec2`/`uvec4` with unsigned 32-bit components and tight stride (8/16). Default mode stays scalar-only. Tests reject the view in default mode, loose stride, `uvec3`, signed and float vectors.
- Iris Xe reports a 4-byte storage offset alignment. The MMQ stage builder rejects scratch that isn't 16-byte aligned, failing closed, and the encoder now aligns every tensor to max(16, device alignment) via the new `VkTensorArena.AllocF32Aligned`.

## Results (5 repeats, medians; every repeat's output identical)

| Fixture | Before | **Wide-load MMQ** | Encoder | Output |
|---|---:|---:|---|---|
| JFK | 3.825 | **3.506** | 3.494→3.213 | same |
| PT row 0 | 3.784 | **3.527** | 3.506→3.218 | same |
| FR row 0 | 3.662 | **3.405** | 3.510→3.245 | same |
| JFK VAD+words | 4.153 | **3.889** | 3.465→3.239 | same |
| groups | 8.344 | **7.857** | 6.916→6.438 | same |
| PT2 | 8.118 | **7.588** | 6.970→6.449 | same |
| podcast | 4.410 | **4.144** | 3.478→3.220 | same |
| podcast VAD+words | 5.462 | **5.220** | 3.444→3.186 | same |
| PT1 | 3.690 | **3.412** | 3.512→3.235 | same |
| silence | 0.021 | 0.021 | — | same |

The native MMQ exactness test passes on 9 shapes, including row and column tails and invalid-marker blocks: Q8_1 words and outputs are bit-identical to the base integer-dot kernel. Rejection of misaligned scratch is exercised on the device. The first run of that test stopped at the new alignment guard because the test allocated unaligned scratch; the test now allocates aligned scratch and asserts the rejection.

## Remaining gap (JFK 3.51 s vs 3.02 s)

| Stage | Go | Original |
|---|---:|---:|
| encoder | 3.20 s | 2.78 s |
| decoder | 0.25 s | ~0.2 s |
| mel | 0.012 s | ~0.01 s |

Encoder targets: attention (fenced 1.76 s vs 1.58 s for flash attention) and FC1/FC2 (~10–15% behind).
