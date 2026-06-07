# model/ideogram4

Ideogram v4 image generation: a DiT (diffusion transformer) with a Qwen-VL
conditioner and a VAE decoder. FP8-weight native path plus GPU (CUDA) variants.

| Area | Files |
|---|---|
| Pipeline / runtime | `native_pipeline.go`, `runtime.go`, `cfg.go`, `config.go`, `denoise_loop.go` |
| DiT | `dit_block.go`, `dit_forward.go`, `latents.go`, `latent_norm.go` |
| Conditioning | `conditioning.go`, `qwen_vl_conditioner.go` |
| Scheduler | `scheduler.go`, `scheduler_runtime.go` |
| FP8 weights | `fp8_layout.go`, `fp8_linear.go`, `fp8_load.go` |
| VAE | `vae_decoder.go`, `vae_ops.go` |
| GPU kernels | `gpu_*.go` (attention, mlp, mrope, norm, VAE conv/groupnorm/upsample, residency) |
| Inventory | `tensor_inventory.go` |

These kernels (Conv2D, GroupNorm, VAE spatial attention, GQA, mRoPE) are
architecture-specific to this diffusion model; shared low-level primitives live in
`backends/simd`. Half-precision conversion is in the `half` package.
