# Ideogram 4 FP8 support

`ideogram-ai/ideogram-4-fp8` is a gated Diffusers text-to-image pipeline released with the `ideogram-oss/ideogram4` reference implementation. It is not a LLaMA/GGUF decoder target; native go-pherence support needs a separate image-generation pipeline.

## Published/local component graph

The gated Hugging Face config reports:

```text
pipeline:                  Ideogram4Pipeline
transformer:               Ideogram4Transformer2DModel
unconditional_transformer: Ideogram4Transformer2DModel
scheduler:                 FlowMatchEulerDiscreteScheduler
text_encoder:              Qwen3VLModel
tokenizer:                 Qwen2Tokenizer
vae:                       AutoencoderKLFlux2
```

## Transformer shape

From the real `ideogram-4-fp8` configs and `ideogram-oss/ideogram4`:

```text
layers=34
emb_dim=4608
num_attention_heads=18
attention_head_dim=256
intermediate_size=12288
in_channels=128
llm_features_dim=53248
rope_theta=5000000
mrope_section=[24 20 20]
```

`llm_features_dim=53248` is `4096 * 13`, from concatenating Qwen3-VL hidden states at activation layers:

```text
[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]
```

## Text encoder

The text encoder config is Qwen3-VL with nested `text_config`:

```text
model_type=qwen3_vl
text.model_type=qwen3_vl_text
hidden_size=4096
num_hidden_layers=36
num_attention_heads=32
num_key_value_heads=8
vocab_size=151936
```

The reference pipeline uses text-only Qwen3-VL hidden-state extraction through the Qwen chat template. Native support therefore needs an encoder-style Qwen3-VL forward path that returns the selected hidden layers, not causal token generation.

## Native implementation requirements

Generation support should be implemented in pure Go with backend-owned kernels and no Python/Diffusers runtime dependency:

1. **Inspection/readiness** — implemented by `loader/config/ideogram4.go`, `model/ideogram4` shape scaffolding, and `cmd/ideogram4inspect` for local Diffusers folders.
2. **Qwen3-VL text conditioning** — initial tokenizer/conditioning shape contracts exist in `model/ideogram4`; the remaining work is Qwen chat-template rendering and native Qwen3-VL forward execution to concatenate the 13 selected hidden states.
3. **Ideogram4 DiT reference path** — implement single-stream text+image token transformer blocks with QK-RMSNorm, MRoPE, SwiGLU MLP, AdaLN timestep conditioning, and velocity prediction.
4. **FP8 weight loading** — tensor-index inventory scaffolding now verifies both conditional and unconditional DiT indexes (`669` tensors each, `34` layers, `211` FP8 weight scales each). The FP8 linear-weight layout contract (`model/ideogram4/fp8_layout.go`) classifies every `.weight`/`.weight_scale` tensor into a linear role with expected in/out dimensions derived from `Config` (`adaln_dim=512`), and `ideogram4inspect` reports per-transformer linear coverage (`required=211 present=211 scaled=211 missing=0`). Remaining work is backend-owned FP8 dequant/GEMV execution.
5. **Unconditional transformer / asymmetric CFG** — run conditional and unconditional paths as in `ideogram-oss/ideogram4`.
6. **FlowMatch Euler scheduler** — initial logit-normal schedule, backward Euler step plan, and asymmetric CFG layout scaffolds exist in `model/ideogram4`; remaining work is wiring them into the DiT execution loop.
7. **AutoencoderKLFlux2 decode** — decode 32-channel latents to image output.
8. **SIMD acceleration** — promote hot reference ops to checked `backends/simd/runtime` APIs and add AVX/NEON/RVV kernels where profiling warrants.
9. **End-to-end fixture** — pin a small prompt/seed/step fixture against the reference implementation before marking runtime ready.

## DiT block forward

`model/ideogram4/dit_block.go` implements one Ideogram4 transformer block natively over the loaded FP8 linears:

- gate-less adaLN modulation (`4*emb` block params split into shift/scale for the attention and MLP sublayers, matching the `2*emb` final-layer modulation contract),
- non-affine LayerNorm + `x*(1+scale)+shift`,
- QKV projection, 3-section MRoPE (`[24,20,20]`, temporal/height/width) over the first `2*sum(section)` head dims via rotate-half,
- full (non-causal) scaled dot-product attention with SIMD softmax,
- output projection residual,
- SwiGLU MLP (`W2(SiLU(W1)*W3)`) residual.

The block math is assembled from the public Ideogram4 tensor contract and config; end-to-end numerical correctness still requires validation against real weights, so the pipeline keeps `runtime_ready=false`.

## Full DiT velocity forward

`model/ideogram4/dit_forward.go` stacks the blocks into a complete velocity pass (`DiTModel.Velocity`):

- `LoadDiTModel` assembles globals + 34 layers from a loaded FP8 set,
- sinusoidal timestep embedding → `t_embedding.mlp_in/out` (SiLU) → `adaln_proj` conditioning,
- `llm_cond_proj`(LayerNorm(text features)) text tokens + `input_proj`(latents) image tokens form one joint self-attention sequence (single qkv/o per block),
- text prefix gets sequential temporal MRoPE positions, image tokens the 2D grid,
- 34 `ForwardLayer` blocks,
- `final_layer.adaln_modulation` + `final_layer.linear` over image tokens → velocity `[imageTokens, in_channels]`.

## Current status

Inspection/runtime scaffolding with a concrete native scheduler, CFG combiner, FP8 E4M3 linear backend, FP8 weight loading, and a native DiT block forward. `FlowMatchScheduler` (weight-free) now natively derives ordered FlowMatch timesteps from the logit-normal schedule and performs the Euler latent update (`x_{t-1} = x_t + sigma * velocity`); `Pipeline.Generate` instantiates it and validates the step plan before the DiT/decode boundary returns not-implemented. `cmd/ideogram4inspect` validates local `ideogram-4-fp8` config folders, converts them into `model/ideogram4.Config`, and reports the actual component graph and dimensions. With optional safetensors index JSONs, it also reports conditional/unconditional transformer tensor inventory, FP8 scale coverage, and FP8 linear-weight role coverage. `model/ideogram4` has bounded prompt-token, text-conditioning, latent shape, FlowMatch schedule, asymmetric CFG layout, tensor-inventory, and FP8 linear-layout helpers, but `runtime_ready=false` until Qwen3-VL conditioning, Ideogram4 DiT execution, FP8 loading, and VAE decode are implemented natively.
