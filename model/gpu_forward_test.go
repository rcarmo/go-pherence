package model

import (
	"testing"

	nvidia "github.com/rcarmo/go-pherence/backends/nvidia/runtime"
)

func TestGPUExpertCachePolicyFromEnv(t *testing.T) {
	t.Setenv("GO_PHERENCE_EXPERT_CACHE_POLICY", "")
	if got := gpuExpertCachePolicyFromEnv(); got != nvidia.ExpertCachePolicyLRU {
		t.Fatalf("default policy=%q want %q", got, nvidia.ExpertCachePolicyLRU)
	}

	t.Setenv("GO_PHERENCE_EXPERT_CACHE_POLICY", "lfu")
	if got := gpuExpertCachePolicyFromEnv(); got != nvidia.ExpertCachePolicyLFU {
		t.Fatalf("lfu policy=%q want %q", got, nvidia.ExpertCachePolicyLFU)
	}

	t.Setenv("GO_PHERENCE_EXPERT_CACHE_POLICY", "broken")
	if got := gpuExpertCachePolicyFromEnv(); got != nvidia.ExpertCachePolicyLRU {
		t.Fatalf("invalid policy fallback=%q want %q", got, nvidia.ExpertCachePolicyLRU)
	}
}
