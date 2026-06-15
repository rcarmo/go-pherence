// Package dgflags reads the shared DiffusionGemma runtime environment toggles
// used by the diffusiongemma command-line tools.
package dgflags

import (
	"os"
	"strconv"
	"strings"
)

func envBool(name string) bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

// ExpertCacheBudgetBytes returns the expert-cache budget in bytes, honoring
// GO_PHERENCE_DIFFUSIONGEMMA_EXPERT_CACHE_MB or the provided default (MB).
func ExpertCacheBudgetBytes(defaultMB int64) int64 {
	if v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_EXPERT_CACHE_MB")); v != "" {
		if mb, err := strconv.ParseInt(v, 10, 64); err == nil && mb >= 0 {
			return mb * 1024 * 1024
		}
	}
	return defaultMB * 1024 * 1024
}

// GPUSelfCondEnabled reports GO_PHERENCE_DIFFUSIONGEMMA_GPU_SELFCOND.
func GPUSelfCondEnabled() bool {
	return envBool("GO_PHERENCE_DIFFUSIONGEMMA_GPU_SELFCOND")
}

// GGUFGPULMHeadEnabled reports GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_LMHEAD.
func GGUFGPULMHeadEnabled() bool {
	return envBool("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_LMHEAD")
}

// GGUFGPULMHeadChunkSize returns the GGUF GPU LM-head chunk size, honoring
// GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_LMHEAD_CHUNK (default 32768).
func GGUFGPULMHeadChunkSize() int {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_LMHEAD_CHUNK"))
	if v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return 32768
}

// GGUFGPULMHeadUseF32Cache reports GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_LMHEAD_F32_CACHE.
func GGUFGPULMHeadUseF32Cache() bool {
	return envBool("GO_PHERENCE_DIFFUSIONGEMMA_GGUF_GPU_LMHEAD_F32_CACHE")
}
