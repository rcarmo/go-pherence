// Package omnivoice loads OmniVoice configuration metadata and validates
// safetensors checkpoint headers against the expected OmniVoice/Qwen3 tensor
// layout.
//
// The package is intentionally header-only for checkpoint inspection: it uses
// loader/safetensors to mmap the file and inspect tensor metadata without
// materializing tensor payloads into Go memory.
package omnivoice
