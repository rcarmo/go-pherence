// Command minicpmvinspect inspects OpenBMB MiniCPM-V and MiniCPM-O
// checkpoints without running full tensor execution.
//
// It reports config, processor, tokenizer/chat-template, generation, image/audio
// sentinel token, prompt placeholder, image preprocessing, safetensor inventory,
// safetensor shape, text/vision/resampler/audio plan, capability, and readiness
// metadata. Readiness flags such as -require-metadata-ready, -require-tensors-ready,
// -require-shapes-ready, and -strict are intended for checkpoint validation before
// numeric MiniCPM-V/O runtime execution lands.
//
// Full generation is intentionally not claimed by this command yet; use
// -require-runtime-ready to fail loudly until the runtime implementations are
// wired and parity-gated.
package main
