# Vulkan Q8 register-tiled linear — 13 September 2026

A 32×32 register-tiled rewrite turns the standalone per-row Q8 weight operator from a slight regression into a repeatable kernel win on Intel Iris Xe. It preserves the stored format and core Vulkan feature contract. This qualifies the operator implementation, not model-level Q8 quality or default placement.

## Optimization

The original 16×16 kernel loaded and decoded one byte for each weight and multiplied every dequantised weight by its row scale before the dot product. Its retained historical result was `0.984×` the current F32 kernel.

The optimized shader:

- computes a 32×32 output tile with one 16×16 workgroup;
- gives each invocation four F32 accumulators/outputs;
- uses an 8KiB shared X/weight tile and K tile 32;
- for word-aligned K, gives each of 256 invocations one packed `uint32` load and expands its four adjacent Q8 values into the complete 32×32 weight tile;
- retains a byte-addressed fallback for odd K;
- accumulates `X×Q8` in F32 and applies the per-output-row F32 scale once after reduction.

The scale-once arithmetic changes the rounding schedule but not the stored representation or quantisation. It is assessed against an F32 oracle using the exact dequantised weights. SPIR-V still declares only `Shader` and 32-bit integer/float types: no `Int8`, 8-bit storage, dot product or subgroup capability.

A rejected cooperative 16×16 variant serialized expansion onto four lanes and measured only `0.903–0.906×` F32. Scale-once on the original tile recovered parity (`0.993–1.002×`) but did not provide a stable win. Both are retained only as diagnostic logs outside the repository.

## Verification

- Exact packing and nearest-even half-tie tests are unchanged.
- A shuffled Go schedule model now mirrors 32×32/four-output accumulation, scale-once arithmetic, packed-word geometry, odd-K fallback and one writer per output.
- Descriptor, alignment, alias, cancellation, in-flight retention and copied-owner tests pass.
- Thirty shuffled focused repetitions pass.
- Affected Vulkan, Whisper and Community-1 packages pass; vet/build pass.
- All 23 stored/embedded/rebuilt shaders validate and rebuild byte-identically.
- Shader source SHA-256: `f00ef2cb2632ca2c8799b34b012e3daae088b35ac8c809cfcb4bb68dde74c8bf`.
- SPIR-V SHA-256: `6ebaec62c644976bc62ad977a9c693d3e805320ace560f94e98247f99863c306`.
- Disassembly SHA-256: `b09fcf051f187d7e5ca9acd61bc611742a9802214bd07fc0ebde2d4326721088`.
- Static verification JSON SHA-256: `1e4a2fc5811ee1a802347bc2680153b00e7d8eb8773024c94b14a0d4a803b4b3`.

## Native numerics and quality

Device: `Intel(R) Iris(R) Xe Graphics (RPL-P)`.

Seven numerical shapes include odd/even K and output edges. Maximum kernel error against the exact dequantised-weight CPU oracle is `2.154904371054478e-7`, below `2e-5 + 2e-5*abs(reference)`. Across the full timing shapes, maximum Q8-versus-F32 error is `3.039836883544922e-6`; the F32 register tile is bit-exact with current F32.

The lossy weight-quality result is unchanged in substance: sampled RMS error/reference-RMS reaches `0.656%` at `K=1280`, with maximum sampled original-F32 absolute error `0.0032398310119566565`. Model-level WER/DER is still required before placement.

## Native timing

Three permanent trials each measured all three projection geometries and three kernels. Every per-shape trial used eight host-wall dispatch samples per kernel in four alternating orders. Input, bias, output and exact dequantised F32 weights are identical. Construction, quantisation, upload and download are excluded. GPU timestamps are unavailable.

Across the three trials:

| Shape | Q8 storage | F32 storage | Q8 vs current F32 | Q8 vs F32 register tile |
|---|---:|---:|---:|---:|
| `1500×1280×1280` | 1,643,520 | 6,553,600 | 2.134–2.144× | 1.032–1.038× |
| `1500×1280×5120` | 6,574,080 | 26,214,400 | 2.226–2.227× | 1.042–1.043× |
| `1500×5120×1280` | 6,558,720 | 26,214,400 | 2.218–2.221× | 1.039–1.041× |

The final resource-measured run reported:

| Shape | Current F32 | F32 register tile | Q8 register tile | Q8/F32 | Q8/F32-register |
|---|---:|---:|---:|---:|---:|
| `1500×1280×1280` | 28.750ms | 13.956ms | 13.452ms | 2.137× | 1.037× |
| `1500×1280×5120` | 115.255ms | 54.041ms | 51.799ms | 2.225× | 1.043× |
| `1500×5120×1280` | 115.106ms | 53.913ms | 51.854ms | 2.220× | 1.040× |

The process completed in `5.67s`, used `163,484KiB` maximum RSS and reported zero process swaps. Host `pswpout` did not change; `pswpin` increased by one page over the run and is reported rather than claimed as zero.

Final native log SHA-256: `34511a2d7f2382b230bcc4abe8996ce670d7a2c13cdf5b5bca6ab36e304e7e00`. Time output SHA-256: `47731944e1b9c0e3f6cdf60f50002130cdfc1bbfb18e09bc453410a9d6db3fba`.

## Decision

The register-tiled Q8 kernel passes its storage, numerical and isolated performance gates. It is now a viable placement candidate, not a rejected kernel. It remains explicit and outside `VkF32Plan`, Whisper, Community-1 and serving defaults until trained model-level quality and whole-graph placement comparisons pass. No tolerance, model hash, provenance or serving path changes here.

## Reproduce

```sh
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/intel_icd.x86_64.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"
export GO_PHERENCE_VULKAN_DEVICE=Iris
GO_PHERENCE_TEST_VULKAN_LINEAR_Q8_TIMING=1 \
  GOMAXPROCS=2 CGO_ENABLED=0 make speech-vulkan-linear-q8-weight-check
```
