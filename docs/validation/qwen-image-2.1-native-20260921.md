# Qwen Image 2.1 native integration — 2026-09-21

This milestone establishes an honest Qwen Image 2.1 boundary in `go-pherence`: native Go config/scheduler/layout/numerics and strict weight admission, plus end-to-end CPU generation through a pinned, source-audited `stable-diffusion.cpp` executable. It does not claim that the 32-layer DiT and specialised VAE execute in pure Go.

## Provenance and licence

- `stable-diffusion.cpp`: MIT, source `c678dfe704a2230342376b46add9c8ca736a653d`; feature commit `137f7409bbfb98c70a350a57d6a135487080db96`.
- Official model: `Qwen/Qwen-Image-2.1` revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`.
- Comfy single-file assets: `ace0edeb3791a594ddfa36ed5f41a178a394e921`.
- Diffusion GGUF: `leejet/Qwen-Image-2.1-GGUF` revision `cc11433936a06e9765f7c0c0b1f0436cfd2b9856`.
- Official weights use the Qwen Research License Agreement: non-commercial research/evaluation only without a separate commercial licence.

Validated asset hashes:

| Asset | SHA-256 |
|---|---|
| `qwen_image_2.1-Q2_K.gguf` | `5f79e41d3424abb230ee1486e1f4adab1b07c8c0fd10601fc402fe9890443fe6` |
| `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | `67d1659bfe71b89d50b45a4ad1a9e5b997e5bb16ce5da66a6a6167abd569e9e2` |
| `qwen_image_2.1_vae_bf16.safetensors` | `bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9` |

## Architecture and native boundary

The model is a 32-layer, 4,096-wide single-stream transformer with 32×128 attention, 12,288-wide SwiGLU, shared four-way modulation, 64-channel unpatched latents, Qwen3-VL 8B final pre-norm hidden states, and a Wan-derived spatial-scale-16 VAE. Attention is causal globally but bidirectional inside each condition/target image block. Text and condition-image tokens use zero-timestep modulation; target tokens use the current timestep.

Native Go covers strict component metadata; image/text/target segment layout; centred 3-axis position coordinates; block-causal admission; dynamic exponential FlowMatch with 0.02 terminal stretch; CFG/Euler steps; zero-centred RMSNorm, LayerNorm, timestep embedding, text projection and modulation; and complete diffusion tensor inventory for the 265-tensor Q2_K file. Package statement coverage is 85.4% without real assets and 88.6% with the pinned GGUF inventory gate, below the preferred 90% target; remaining misses are process and malformed-loader branches. Actual DiT/Qwen3-VL/VAE tensor execution uses the pinned MIT C++ backend through argv-based process execution with timeout, CPU-only environment, PNG validation and hash reporting.

## End-to-end proof

CPU-only command contract: prompt `a red square on a white background`, Euler, one step, CFG 1.0, seed 42, 32×32, six threads. No GPU device was detected or used.

- Direct pinned `sd-cli`: generation 2.84 s; process wall 3.39 s; peak RSS 7,778,212 KiB.
- Go `cmd/image/qwenimage21`: generation 2.80 s; process wall 3.59 s; peak RSS 7,788,636 KiB.
- Both outputs are byte-identical PNGs: SHA-256 `5a129ef0587d326aaf6aa55942b0289f3e276ebf190f5709a97bf66fa171bfe0`, 3,456 bytes.

This is proof of executable integration, not a quality claim. At 32×32 and one denoising step the output is only a low-resolution smoke artifact. High-resolution quality and image editing are not qualified; editing also needs Qwen3-VL vision weights.

Final gates passed: 120 race-tested packages plus 55 no-test packages, `go vet ./...`, host build, Linux ARM64 and RISC-V builds, 382-document link/build checks, and native ARM64 package execution. The frozen experiment remains unchanged at 16 manifest entries, two binaries and 676 records.
