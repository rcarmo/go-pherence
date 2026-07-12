# Runtime tuning contract

This document defines the common, intentionally small tuning surface for go-pherence. It is not a promise of command-line compatibility with `llama.cpp`: a similarly named control is accepted only when go-pherence can give it a documented meaning.

## Common controls

| Area | Control | Meaning | Availability |
|---|---|---|---|
| Placement | `gpu` | Select CPU or the compiled GPU backend | `llmgen`, `llmchat`, `llmserver` |
| Placement | `gpu-layers` / `ngl` | Number of transformer layers assigned to GPU; `0` keeps layers on CPU | LLM commands and server presets |
| Placement | `gpu-kv-max-seq` | Maximum GPU KV sequence allocation | `llmgen`; `GO_PHERENCE_GPU_KV_MAX_SEQ` elsewhere |
| Execution | `threads` | Execution hint; it does not reproduce llama.cpp thread-pool scheduling | `llmserver` preset/CLI |
| Execution | `batch-size` | Execution hint; it does not reproduce llama.cpp physical/ubatch scheduling | `llmserver` preset/CLI |
| Context | `ctx-size` | Maximum request context admitted by the server | `llmserver` preset/CLI |
| KV policy | `cache-type-k`, `cache-type-v` | Select native full-precision or TurboQuant key/value storage | GGUF generation/server |
| KV policy | `kv-residual-window` | Keep the newest N KV positions uncompressed | GGUF generation/server CLI |
| Generation | `max_tokens` | Maximum generated tokens; zero uses the server default | OpenAI-compatible server API |
| Generation | `temperature=0` | Greedy argmax decoding | OpenAI-compatible server API |

Programmatic entry points are `model.LoadGPUModelWithLayers` for placement and `model.GGUFGenerationOptions` plus `runtime/kv.TurboQuantConfigFromCacheTypes` for GGUF KV policy. DiffusionGemma has a separate algorithm-specific contract through `diffusiongemma.InferenceOptions` and `diffusiongemma.DenoisingConfig`; its canvas, denoising, and residency controls are not interchangeable with autoregressive LLM controls.

## KV cache semantics

Accepted full-precision names are empty, `none`, `full`, `f16`, `bf16`, and `f32`. Accepted quantized aliases are `turbo2` through `turbo4` and the supported `q*_k`/`q*_0` names listed by `runtime/kv.ParseCacheTypeBits`.

These aliases select go-pherence's native TurboQuant implementation. They do **not** select llama.cpp's cache tensor formats and do not promise byte-for-byte or numerical equivalence. See [TurboQuant](turboquant.md).

## Explicitly unsupported llama.cpp controls

The following controls are not silently approximated:

| llama.cpp control family | go-pherence behavior |
|---|---|
| `flash-attn` | Unsupported; graph/backend selection is automatic and backend-specific |
| `split-mode`, `tensor-split`, `main-gpu` | Unsupported; only coarse single-backend `gpu-layers` placement is available |
| Nonzero `temperature` | Unsupported for autoregressive generation; requests return HTTP 400 |
| `top-k`, `top-p`, `min-p`, `typical-p` | Unsupported for autoregressive generation |
| `mirostat` | Unsupported |
| `repeat-penalty`, `presence-penalty`, `frequency-penalty` | Unsupported |
| `seed` | Unsupported for greedy autoregressive generation; DiffusionGemma has its own seed |
| llama.cpp diffusion algorithms, block count, CFG, Gumbel noise, epsilon, and diffusion KV tri-state | Unsupported; use the documented DiffusionGemma denoising controls instead |

Known unsupported placement and sampling keys in `llmserver` model presets return a configuration error with the key and line number. Unknown unrelated llama-server keys remain ignored so a shared preset can retain metadata and server-only options. JSON API fields are decoded strictly; unknown request controls return HTTP 400.

## Intentional differences

- TurboQuant cache formats are native go-pherence formats, even when a llama.cpp-style alias is accepted.
- Backend placement is a coarse CPU/GPU layer count, not multi-GPU tensor splitting.
- DiffusionGemma implements its own canvas denoising schedule and sampling API rather than llama.cpp diffusion algorithm IDs.
- Speculative decoding controls configure proposal/verification, not token sampling.
