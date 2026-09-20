# DiffusionGemma FP8 end-to-end CPU validation -- 2026-09-20

The local `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic` checkpoint now runs through native Go safetensor loading, indexed FP8 experts, prompt encoding, all 30 text layers, block denoising, sampling, tokenizer decoding, the OpenAI-compatible HTTP server and the embedded llama.cpp UI. No GPU was queried or used.

This is bounded text-path execution evidence, not full runtime readiness or a quality benchmark. Capability metadata still reports `runtime_ready=false`: broader reference parity and full image-sequence vision fixtures remain missing. The vision path is outside this validation.

## The missing binding

The FP8 checkpoint stores each expert separately (`layers.<L>.experts.<E>.{gate,up,down}_proj`) rather than in fused three-dimensional expert tensors. `OpenTextWeights` correctly marked `IndexedExperts`, and `BuildFP8ExpertIndex` already resolved every weight/scale pair, but the CPU runner and HTTP server constructed `CPUDispatcher` without that index. Real execution failed before layer math:

```text
DiffusionGemma encoder: DiffusionGemma encoder expert tensor bindings missing layer 0
```

`OpenCPUExpertIndex` now centralizes setup. Fused checkpoints return no index and keep their existing binding path. Indexed checkpoints open FP8 weights, build the full index, return explicit ownership to the caller and fail startup transactionally if any expert is absent or malformed. Both `diffusiongemmarun` and `diffusiongemmaserver` retain and close that mapping for the dispatcher lifetime. The server now also passes configured final-logit softcapping to its CPU dispatcher, matching the runner.

A real checkpoint test builds all **30 layers × 128 experts** and checks hidden/intermediate geometry. On the local host, index construction itself logged 21--104 ms across bounded runs; the isolated `go test` process peaked around 4.43 GiB RSS, including Go build/test overhead and mappings.

## Checkpoint and bounded execution

The local model directory contains a single 27,200,864,632-byte safetensor plus its index, config, generation defaults, tokenizer and chat template. Metadata inspection completed in 0.53 seconds at roughly 89 MiB RSS and reported:

* 30 text layers, hidden width 2,816;
* 128 experts, top 8, intermediate width 704;
* vocabulary 262,144 and canvas 256;
* all 18 operations implemented, 15 reference-complete;
* complete shard inventory (1/1).

The first one-layer CPU smoke (`canvas=1`, one denoising step, no tail) completed after the index fix in 8.1 seconds at about 10.6 GiB RSS. It returns token zero because the debug path intentionally omits the remaining layers and LM tail; that output is plumbing evidence only.

A full 30-layer one-token run completed in 8.2 seconds at about 13.7 GiB RSS but used raw prompt ID 2 and trimmed immediate EOG. A tokenizer prompt without the generation header did the same. This established that prompt formatting matters rather than providing useful text.

With the checkpoint's chat template and model generation header, one full four-token canvas and one denoising step produced:

```text
IDs:  [2088, 236888, 2088, 740]
Text: " How! How can"
```

The 11-token prompt was `<|turn>user ... <|turn>model` plus the disabled-thinking channel header. Wall time was 15.5 seconds and maximum RSS about 16.8 GiB. A second independent process produced the same four IDs exactly. One denoising step is deliberately small and does not establish response quality.

Using the default 48-step schedule with a four-token user-only prompt completed in 87.1 seconds and peaked around 24.5 GiB RSS, but omitted the model generation header and trimmed to EOG. It is retained only as a resource measurement, not output evidence.

## HTTP and UI

`diffusiongemmaserver` was built and launched locally with the same FP8 checkpoint, NVIDIA disabled, two Go threads, canvas four, one denoising step and all text layers. Startup built the expert index and served:

* `/healthz`;
* `/props` with model/context/default generation settings;
* `/v1/models`;
* `/v1/chat/completions`;
* embedded llama.cpp UI at `/` with HTTP 200.

A non-streaming OpenAI chat request for `Hello` returned:

```json
{"choices":[{"message":{"role":"assistant","content":" Hello!"},"finish_reason":"stop"}],"usage":{"prompt_tokens":13,"completion_tokens":2,"total_tokens":15}}
```

The request took 13.74 seconds; prompt KV encoding logged 8.6 seconds and effective output throughput was about 0.146 tokens/s. This proves route, tokenizer/template, engine and response integration. It does not make the CPU path production-ready. Server startup may take several seconds; health polling observed transient connection failures before listening.

## Reproduction

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go run ./cmd/diffusiongemmainspect \
  -model checkpoints/diffusiongemma-26B-A4B-it-FP8 -json

GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 \
  go run ./cmd/diffusiongemmarun \
  -model checkpoints/diffusiongemma-26B-A4B-it-FP8 \
  -cpu-dispatcher -allow-slow-cpu \
  -message user:Hello -generation-prompt -decode \
  -max-new 4 -canvas 4 -denoise-steps 1 -tail-after-max-layers -json

GO_PHERENCE_DISABLE_NVIDIA=1 GOMAXPROCS=2 \
  go run ./cmd/diffusiongemmaserver \
  -model checkpoints/diffusiongemma-26B-A4B-it-FP8 \
  -listen 127.0.0.1:18092 -webui -allow-slow-cpu \
  -max-new 4 -canvas 4 -denoise-steps 1 -tail-after-max-layers
```

The `-allow-slow-cpu` gate is intentional. The server admits one inference request at a time and returns 429 when busy. Existing request/body/context bounds remain unchanged.

## Tests, limits and remaining work

Affected DiffusionGemma/server race tests pass, including FP8 projection/expert, generation, graph and HTTP/UI tests. Final whole-tree, documentation, cross-build and preservation gates are recorded at publication.

Outstanding work remains substantial:

* text capability is still not broadly reference-complete;
* full vision-sequence reference parity remains absent;
* no GPU qualification was performed;
* CPU throughput and peak RSS are unsuitable for routine service use;
* no released-task quality suite or multi-step response assessment was run;
* GGUF golden tests still require an unavailable local GGUF asset;
* the HTTP validation is local and single-request, not a concurrency/soak test.

The stale “denoiser not implemented” comments are corrected: CPU/SIMD text math is implemented and exercised, while formal runtime readiness remains false for the reference gaps above. Independent delegated review timed out and supplies no review coverage.
