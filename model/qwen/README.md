# model/qwen

Qwen3 / Qwen3.5 inference, including the native multi-token-prediction (MTP) path
and a GPU (CUDA/MLX) execution path. Builds on the GGUF core in the parent
`model` package.

| Area | Files |
|---|---|
| Core | `qwen35.go`, `qwen35_source.go`, `qwen35_bundle.go`, `qwen35_load_helpers.go`, `ops.go`, `schedule.go` |
| Linear / quant / rope | `qwen35_linear.go`, `qwen35_quant.go`, `qwen35_rope.go` |
| Prompt cache | `prompt_cache.go`, `qwen35_gpu_prompt_cache.go`, `qwen35_gpu_cache.go` |
| GPU state | `qwen35_gpu_state.go`, `qwen35_gpu_state_bf16.go`, `qwen35_mlp_gpu.go`, `qwen35_mlx_gpu.go` |
| MTP | `qwen_native_mtp.go`, `qwen_native_mtp_safetensors.go`, `qwen_native_mtp_synthetic.go` |
| Validation | `qwen35_validate_helpers.go` |

Numeric helpers (`rmsNormInPlace`, `gemvNT`, `applyRoPEPartial`) are thin wrappers
over `backends/simd`; bf16 conversion uses the `half` package.
