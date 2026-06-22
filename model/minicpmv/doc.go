// Package minicpmv provides MiniCPM-V/O metadata, prompt, preprocessing,
// tensor-inventory, and readiness scaffolding.
//
// The implemented surface is intentionally explicit about what is ready today:
// config/processor/tokenizer/generation sidecar parsing, image/audio special
// token resolution, image/audio/multimodal prompt placeholder construction,
// image preprocessing, safetensor header inventory/shape checks, text/vision/
// resampler/audio execution planning, aggregate metadata loading, readiness
// reports, and not-implemented runtime interfaces.
//
// Full numeric tensor execution is not implemented yet. The VisionTower,
// Resampler, TextBackbone, and AudioEncoder interfaces plus
// ErrRuntimeNotImplemented define the future runtime boundary while preventing
// the metadata scaffold from accidentally claiming generation readiness.
//
// Validate the scaffold from the project root with:
//
//	make minicpmv-check
package minicpmv
