# Qwen Image 2.1

`model/qwenimage21` adds strict Qwen Image 2.1 contracts, native CPU/SIMD primitives, and a bounded end-to-end runner for the audited MIT-licensed [`stable-diffusion.cpp`](https://github.com/leejet/stable-diffusion.cpp) backend.

Pinned provenance:

- `stable-diffusion.cpp` source `c678dfe704a2230342376b46add9c8ca736a653d`; Qwen Image 2.1 feature commit `137f7409bbfb98c70a350a57d6a135487080db96`.
- Official `Qwen/Qwen-Image-2.1` revision `b3179ad355be050328e483a9dfdd9e60cd62adfa`.
- Comfy single-file assets revision `ace0edeb3791a594ddfa36ed5f41a178a394e921`.
- Diffusion GGUF revision `cc11433936a06e9765f7c0c0b1f0436cfd2b9856`.

The official model is under the **Qwen Research License Agreement**: non-commercial research/evaluation only unless a separate commercial licence is obtained. The Go package and backend integration do not relicense the model weights.

## Native surface

- Strict Diffusers component/config validation for `QwenImage21Pipeline`, the 32-layer 4096-wide joint transformer, dynamic FlowMatch scheduler and 64-channel Wan-derived VAE.
- Exact text/reference/target segment layout, centred 3-axis positions, target metadata and block-causal attention relation.
- Dynamic exponential FlowMatch schedule with terminal stretch, Euler update and CFG combination.
- Zero-centred RMSNorm, LayerNorm, timestep embedding, text projection and prefix/target modulation primitives using shared SIMD operations where available.
- Strict GGUF/safetensor diffusion inventory checks for all 265 required tensors.
- `Runner.Generate` executes a caller-supplied pinned `sd-cli` with argv (not shell text), a bounded timeout, CPU-only environment, required assets, PNG validation and output SHA-256.

```go
r := qwenimage21.Runner{
    Executable: "/path/to/sd-cli",
    Assets: qwenimage21.Assets{
        DiffusionModel: "/path/to/qwen_image_2.1-Q2_K.gguf",
        VAE: "/path/to/qwen_image_2.1_vae_bf16.safetensors",
        LLM: "/path/to/Qwen3VL-8B-Instruct-Q4_K_M.gguf",
    },
}
result, err := r.Generate(ctx, qwenimage21.GenerateOptions{
    Prompt: "a red square on a white background",
    Output: "out.png", Width: 32, Height: 32,
    Steps: 1, Guidance: 1, Seed: 42, Threads: 6,
})
```

The equivalent CLI is `go run ./cmd/image/qwenimage21 ...`. `-inspect` validates diffusion weight readiness without generation.

## Limits

Package statement coverage is 85.4% without real assets and 88.6% with the pinned GGUF inventory check, below the preferred 90% target; remaining misses are process and malformed-loader branches. The end-to-end path currently delegates tensor execution to the pinned C++/GGML backend; it is not a pure-Go DiT/VAE implementation. Text-to-image is validated; image editing remains outside the Go runner because it also requires Qwen3-VL vision weights and reference-image transport. Full-quality/high-resolution evaluation is not claimed from the 32×32 one-step proof. No GPU was used.
