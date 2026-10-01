# Integer-dot projections on the original MMQ schedule — 1 October 2026

A source-guided rewrite of the explicit Q5_0×Q8_1 projection kernel makes the JFK request **15.7% faster** (5.230→4.409 s) than the existing all-projection integer-dot path, while keeping every encoder hidden value bit-identical. Combined with integer-dot for all encoder projections, the JFK request is 4.41 s against 6.13 s for the padded exact-F32 path and 3.02 s for pinned whisper.cpp. The speed target remains unmet, and the Portuguese/JFK timestamp differences of the integer-dot modes remain unqualified. Defaults, services and resources are unchanged.

## Why the original was faster

The pinned original profiler logs (`GGML_VK_PERF_LOGGER`, [attribution](whisper-original-attribution-20261001.md)) list per-operation device time for the JFK encoder. Matching them against the Go stage profile of `vulkan-original-q5-padded-integer-dot-all` gives:

| JFK encoder pass | whisper.cpp device time | Go all-intdot, fenced stages |
|---|---:|---:|
| 192 projections, including Q8_1 quantisation | 1.027 s | 2.42 s |
| Flash attention | 1.580 s | 1.766 s |
| Add/norm/GELU/copy/mul/convolution | ~0.29 s | ~0.33 s |
| Decoder cross K/V | 0.023 s (GPU) | CPU SGEMM |
| Decoder, per token | ~9 ms (GPU `MUL_MAT_VEC q5_0`) | CPU; ~0.72 s phase total for JFK |

Fenced Go stages include per-submission overhead and overstate small stages; they are attribution, not request timings.

Source reading at `c44b60b8` shows the projection difference is in scheduling, not arithmetic:

- `ggml_vk_get_mul_mat_mat_pipeline` always selects `f32acc` for Q8_1 MMQ. `mul_mmq_funcs.glsl` evaluates `d_a*(q_sum*d_b-16*s_b)` per block. Go's existing integer-dot kernel already used the same block arithmetic.
- On Intel Xe without cooperative matrices, `mul_mat_l` is disabled, so the medium int warptile applies: 128 invocations, 64×64 tile, 32 outputs per invocation, `BK_STEP=4` (four 32-value blocks per shared-memory stage), and Q5 unpacking in registers.
- Go's existing kernel used 256 invocations, 16 outputs each, one block per barrier pair, and unpacked Q5 bits in the shared-memory load loop.

Flash attention is called without `ggml_flash_attn_ext_set_prec`, so the encoder uses `GGML_PREC_DEFAULT`. On this fp16-capable device that means F16 K/V and the F16-accumulator scalar flash-attention variant (Intel: 128 invocations, `Br=4`, `Bc=32`, `D_split=8`, subgroups disabled). Go attention is F32 throughout. That is the next attention and compatibility target.

## Changes

- `shaders/integer-dot/linear-mmq.glsl`: the original MMQ schedule with Go's existing per-output arithmetic. Each output keeps integer block sums, `precise` `dot*ds.x-16*ds.y`, increasing-block FMA accumulation and bias added last. Requires K%128==0; `NewVkLinearQ5IntegerDotMMQSetStream` rejects other shapes before reading weights. There is no fallback.
- `shaders/integer-dot/q8-coop.glsl`: the same Q8_1 words as `q8.glsl`, computed by eight invocations per block with coalesced loads. Maximum, invalid flags and integer sums are order-independent.
- Private encoder modes and benchmark backends, not public constructors:
  - `vulkan-original-q5-padded-integer-dot`: padded key extent plus the existing integer-dot FFN.
  - `vulkan-original-q5-padded-integer-dot-all`: Q/K/V/O/FC1/FC2 from original Q5 blocks with Q8_1 activations. Native weights fall from 1,184.9 MB to 503.3 MB.
  - `vulkan-original-q5-padded-integer-dot-mmq`: the same, on the MMQ schedule.
- Stage profiler admission for the `-all`/`-mmq` modes, with Q8 stages labelled.
- The integer-dot shader gate now covers four shaders. Both new shaders regenerate byte-for-byte with shaderc v2026.1, validate, and match their test-data fixtures. The SPIR-V admission checker accepts both without widening; the MMQ shader uses the same typed `OpSUDot` as the existing kernel.

## Exactness

| Check | Result |
|---|---|
| Native operator, 9 shapes (1×128×1 … 1500×5120×1280; row/feature tails, zero block, infinity, >1000, tiny-maximum markers), 5 repeats | Q8_1 words and outputs bit-identical to the base kernels |
| Full padded encoder, pinned turbo Q5, 1,920,000 hidden values | Bit-identical to `-all`; stats unchanged; cancellation at 3 of 1305 checkpoints, drain/reuse and leak checks pass |
| JFK/PT requests, 5 repeats | Outputs identical to `-all` |

Operator timings, mean of 5 alternating runs including quantisation: 1500×1280→1280 7.7→5.1 ms; →5120 23.7→13.6 ms; 5120→1280 23.7→16.4 ms. The cooperative quantiser does not change request time; its stage-profile cost was mostly fence overhead.

## Request results

Five sequential repeats per arm, including the first. Arms ran in sequence, not interleaved.

| Fixture | Padded exact F32 (earlier pair) | Padded + intdot FFN | Padded + intdot all | Padded + intdot all, MMQ |
|---|---:|---:|---:|---:|
| JFK noVAD | 6.133 | 5.604 | 5.230 | **4.409 / 4.439** |
| PT row0 noVAD | 6.103 | 5.560 | 5.204 | **4.409 / 4.405** |
| FR row0 noVAD | 5.691 | 5.158 | 4.819 | — |
| JFK VAD + words | 6.873 | 6.331 | 5.974 | — |
| Two JFK groups VAD + words | 13.856 | 12.769 | 12.057 | — |

MMQ values are two separate five-repeat runs (MMQ kernel, then MMQ plus cooperative quantiser). Every arm repeats its own outputs exactly.

The integer-dot modes keep the earlier timing failures. PT row0 ends at 9.04 s; the original ends at 7.36 s and the exact paths at 7.36–7.38 s. JFK ends at 11.0 s against 10.40 s. Padded FFN-only and all-projection outputs are identical on nine of ten fixtures; podcast segmentation differs. Since the original uses the same MMQ block arithmetic, the remaining mismatch most likely involves its F16 flash attention or K/V storage. That still has to be measured; it is not established here.

## Verification and isolation

`make model-layout-check host-build host-vet host-test docs-check` (537 Markdown files, zero broken links), race tests for `backends/vulkan` and `model/whisper`, and ARM64/RISC-V builds pass. A first gate run failed in `speechjobserve` because the runner exported `WHISPER_THREADS`, which that serving profile rejects. The rerun unset it, matching earlier gate runs; no code changed in between.

All native runs used the @llama isolation hold: CPU4/8GiB/no-swap, physical Intel Iris Xe, Qwen idle, host available memory ≥6 GiB, native deadlines ≤120 s. All containers exited 0 without OOM and were removed.

[Hashed evidence](../../benchmarks/speech-foundations/whisper-intdot-mmq-20261001/) holds environments, request JSON, stage profiles, native logs/states/exits/monitors, shader gate output, parsers and provenance. Test binaries are recorded by hash only.

## Next

1. Port the original scalar flash-attention configuration (F16 K/V, F16 accumulation, Intel tuning) as an explicit mode. Measure speed and test whether it moves PT/JFK endpoints to the original's.
2. Move cross K/V and decoder projections to the GPU (original: ~9 ms per token).
3. Tune projections further (1.3–1.5 s in-graph versus 1.03 s).
