# BF16 parity expectations

BF16 paths are split across CPU SIMD/reference helpers and NVIDIA runtime wrappers. This page records which pairings are expected to match closely and which are deliberately backend-only until a model path requires more.

## Covered runtime paths

| Primitive | CPU owner | NVIDIA owner | Parity expectation |
|---|---|---|---|
| BF16 RMSNorm with weight | `backends/simd/quant/bf16.BF16RMSNorm` via runtime facade | `backends/nvidia/runtime.DevBF16RMSNorm` and native fallback wrapper | Match after BF16 rounding for synthetic vectors. Native BF16 may use hardware rounding; tolerate BF16 last-bit differences when compared through F32. |
| BF16 RMSNormNoScale | N/A as a BF16 CPU helper today; F32 no-scale RMSNorm lives in `backends/simd/runtime` | `backends/nvidia/runtime.DevBF16RMSNormNoScale` | NVIDIA-only helper for BF16 buffers. CPU parity should be added if a CPU BF16 no-scale model path appears; current CPU model no-scale path uses F32 wrappers. |
| BF16 VecAdd | `backends/simd/quant/bf16.BF16VecAdd` via runtime facade | `backends/nvidia/runtime.DevBF16VecAdd` and native fallback wrapper | Match after BF16 rounding for synthetic vectors. |
| BF16 fused activations | CPU activation owner is F32 scalar under `backends/simd/kernels`; BF16 CPU activation helper is not a public model path | `DevBF16SiLUMul`, `DevBF16GELUTanhMul` | GPU BF16 helpers are backend-owned fast paths; compare against F32 scalar activation rounded to BF16 when smoke-testing on hardware. |
| BF16 LM head | CPU path is model F32/quantized LM-head fallback | `UploadBF16LMHead` + `BF16LMHeadWithBuffer` | NVIDIA smoke should compare logits against CPU F32 materialization for small vocab/head fixtures. Buffer byte-size and overflow guards are covered before launch. |

## Current decision

No additional BF16 no-scale CPU helper is required for current model behavior. The relevant production CPU path uses F32 `RMSNormNoScale`; the NVIDIA BF16 no-scale wrapper exists for BF16 GPU buffers and should be smoke-tested on hardware when NVIDIA validation is available.
