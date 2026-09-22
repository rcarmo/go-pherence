// Package minicpmv provides MiniCPM-V/O metadata, prompt, preprocessing,
// tensor-inventory, and correctness-first CPU execution components.
//
// Implemented surfaces include config/processor/tokenizer/generation sidecar
// parsing, image/audio special-token and prompt planning, image preprocessing,
// safetensor inventory/shape checks, owned-F32 MiniCPM/Qwen2/Mistral text,
// nested/fused-QKV SigLIP vision, perceiver resampling, MiniCPM-O Whisper audio,
// and non-aliasing image/audio embedding injection.
//
// RuntimeStatus remains pending until independent released-model text, vision,
// resampler, and audio parity plus end-to-end multimodal generation gates pass.
// Stable runtime interfaces and ErrRuntimeNotImplemented preserve explicit
// boundaries for stages a caller has not bound.
//
// Validate the package from the project root with:
//
//	make minicpmv-check
package minicpmv
