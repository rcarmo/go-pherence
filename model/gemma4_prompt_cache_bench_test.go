package model

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/promptcache"
)

type gemma4PromptCacheBenchCase struct {
	name       string
	warmPrompt []int
	prompt     []int
}

func BenchmarkGemma4PromptCacheColdFullHitPartialHit(b *testing.B) {
	m := newGemma4PromptCacheTestModel()
	basePrompt, divergentPrompt, _ := gemma4PromptCacheSyntheticPrompts()
	cases := []gemma4PromptCacheBenchCase{
		{name: "cold", prompt: basePrompt},
		{name: "full_hit", warmPrompt: basePrompt, prompt: basePrompt},
		{name: "partial_hit", warmPrompt: basePrompt, prompt: divergentPrompt},
	}
	for _, blockSize := range []int{2, 4} {
		for _, tc := range cases {
			b.Run(fmt.Sprintf("synthetic/block_%d/%s/prefill", blockSize, tc.name), func(b *testing.B) {
				benchmarkGemma4PromptCachePhase(b, m, gemma4TestCacheIdentity(fmt.Sprintf("synthetic-%d-%s", blockSize, tc.name)), blockSize, 1<<20, tc, false)
			})
			b.Run(fmt.Sprintf("synthetic/block_%d/%s/decode1", blockSize, tc.name), func(b *testing.B) {
				benchmarkGemma4PromptCachePhase(b, m, gemma4TestCacheIdentity(fmt.Sprintf("synthetic-%d-%s", blockSize, tc.name)), blockSize, 1<<20, tc, true)
			})
		}
	}
}

func BenchmarkGemma4PromptCacheColdFullHitPartialHitRealE4BGGUF(b *testing.B) {
	if os.Getenv("GO_PHERENCE_GEMMA4_PROMPT_CACHE_REAL") == "" {
		b.Skip("set GO_PHERENCE_GEMMA4_PROMPT_CACHE_REAL=1 to run the local Gemma4 E4B GGUF prompt-cache benchmark")
	}
	path := gemma4PromptCacheRealBenchPath(b)
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		b.Fatalf("load Gemma4 GGUF: %v", err)
	}
	basePrompt, divergentPrompt := gemma4PromptCacheRealBenchPrompts()
	cases := []gemma4PromptCacheBenchCase{
		{name: "cold", prompt: basePrompt},
		{name: "full_hit", warmPrompt: basePrompt, prompt: basePrompt},
		{name: "partial_hit", warmPrompt: basePrompt, prompt: divergentPrompt},
	}
	for _, blockSize := range []int{2, 4} {
		prepared := m.PreparedGenerateTokens(basePrompt)
		if len(prepared) < 8 || len(prepared)%blockSize != 0 {
			b.Fatalf("prepared prompt len=%d unsuitable for block size=%d", len(prepared), blockSize)
		}
		for _, tc := range cases {
			b.Run(fmt.Sprintf("real/block_%d/%s/prefill", blockSize, tc.name), func(b *testing.B) {
				benchmarkGemma4PromptCachePhase(b, m, gemma4PromptCacheRealBenchIdentity(path), blockSize, 1<<30, tc, false)
			})
			b.Run(fmt.Sprintf("real/block_%d/%s/decode1", blockSize, tc.name), func(b *testing.B) {
				benchmarkGemma4PromptCachePhase(b, m, gemma4PromptCacheRealBenchIdentity(path), blockSize, 1<<30, tc, true)
			})
		}
	}
}

func benchmarkGemma4PromptCachePhase(b *testing.B, m *LlamaModel, identity promptcache.Identity, blockSize int, budget int64, tc gemma4PromptCacheBenchCase, decodeOnly bool) {
	b.Helper()
	b.ReportAllocs()
	var totalUsedBytes float64
	var totalHits float64
	var totalMisses float64
	var totalEvictions float64
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		cache := promptcache.New(budget)
		cfg := Gemma4PromptCacheConfig{Cache: cache, Identity: identity, BlockSize: blockSize}
		if tc.warmPrompt != nil {
			mustWarmGemma4PromptCache(b, m, cfg, tc.warmPrompt)
		}
		s, err := NewGemma4DecodeSessionWithPromptCache(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 1}, cfg)
		if err != nil {
			b.Fatalf("NewGemma4DecodeSessionWithPromptCache: %v", err)
		}
		before := cache.Stats()
		var after promptcache.Stats
		if decodeOnly {
			if _, err := s.PrefillChunk(tc.prompt); err != nil {
				s.Close()
				b.Fatalf("PrefillChunk(%v): %v", tc.prompt, err)
			}
			after = cache.Stats()
			b.StartTimer()
			if _, err := s.DecodeStep(); err != nil {
				b.StopTimer()
				s.Close()
				b.Fatalf("DecodeStep(%v): %v", tc.prompt, err)
			}
			b.StopTimer()
		} else {
			b.StartTimer()
			if _, err := s.PrefillChunk(tc.prompt); err != nil {
				b.StopTimer()
				s.Close()
				b.Fatalf("PrefillChunk(%v): %v", tc.prompt, err)
			}
			b.StopTimer()
			after = cache.Stats()
		}
		if err := s.Close(); err != nil {
			b.Fatalf("Close: %v", err)
		}
		totalUsedBytes += float64(after.UsedBytes)
		totalHits += float64(after.Hits - before.Hits)
		totalMisses += float64(after.Misses - before.Misses)
		totalEvictions += float64(after.Evictions - before.Evictions)
	}
	b.ReportMetric(totalUsedBytes/float64(b.N), "cache_used_bytes")
	b.ReportMetric(totalHits/float64(b.N), "cache_hits")
	b.ReportMetric(totalMisses/float64(b.N), "cache_misses")
	b.ReportMetric(totalEvictions/float64(b.N), "cache_evictions")
}

func gemma4PromptCacheRealBenchPath(b *testing.B) string {
	b.Helper()
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		root := findMTPGraphBenchRepoRoot()
		path = filepath.Join(root, "models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
	}
	if _, err := os.Stat(path); err != nil {
		b.Skipf("local Gemma4 GGUF unavailable: %v", err)
	}
	return path
}

func gemma4PromptCacheRealBenchIdentity(path string) promptcache.Identity {
	return promptcache.Identity{
		ModelFingerprint:      "gemma4-e4b-gguf",
		CheckpointFingerprint: path,
		Backend:               string(InferenceBackendSIMD),
		WeightLayout:          "gguf",
		WeightDType:           "q4_0",
		KVPolicy:              "full",
		KVPrecision:           "f32",
		ConfigFingerprint:     "gemma4-e4b",
		RoPEFingerprint:       "gemma4",
		CacheSalt:             "benchmark-real",
	}
}

func gemma4PromptCacheRealBenchPrompts() (base, divergent []int) {
	return []int{10979, 236764, 10979, 236764, 10979, 236764, 10979}, []int{10979, 236764, 10979, 236764, 10979, 236789, 236757}
}
