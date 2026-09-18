# DiffusionGemma vision boundary

The multimodal implementation reads processor/tokenizer metadata, expands image placeholders, preprocesses images in the Gemma4 style, and plans vision tensor bindings. CPU prefix execution and one-layer-at-a-time streaming tower entry points exist, along with the projection/insertion boundary for image embeddings.

These pieces do not yet establish full image-conditioned generation parity. `model/diffusiongemma/reference_gaps.go` retains the full image-sequence reference-fixture gap; overall readiness stays false even when text generation succeeds.

The full streaming tower defaults to a 64-patch safety limit. `GO_PHERENCE_DIFFUSIONGEMMA_VISION_FULL_STREAMING_MAX_PATCHES` explicitly overrides that guard. Raising it permits a larger validation experiment; it does not turn the unvalidated path into a supported full-image pipeline. Capabilities report the effective limit and whether the override parses correctly.

Use the [historical vision investigations](../../history/diffusiongemma/implementation-log.md) to reproduce bounded experiments, and record image preprocessing, patch count, tensor revision and numerical comparisons when extending them. A successful prefix or an allocated embedding tensor is not a full tower reference test.

[Text support](README.md) | [Runtime](runtime.md) | [Validation](validation.md)
