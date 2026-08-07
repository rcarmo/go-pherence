package model

import (
	"fmt"
	"sync"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/promptcache"
)

func newGemma4PromptCacheTestModel() *LlamaModel {
	m := newSingleLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	return m
}

func gemma4PromptCacheSyntheticPrompts() (base, divergent, disjoint []int) {
	return []int{0, 1, 2, 0, 1, 2, 0, 1}, []int{0, 1, 2, 0, 1, 2, 1, 0}, []int{2, 2, 1, 1, 2, 2, 1, 1}
}

func mustRunGemma4PromptCacheSession(tb testing.TB, m *LlamaModel, cacheCfg Gemma4PromptCacheConfig, prompt []int, maxTokens int) []int {
	tb.Helper()
	s, err := NewGemma4DecodeSessionWithPromptCache(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: maxTokens}, cacheCfg)
	if err != nil {
		tb.Fatalf("NewGemma4DecodeSessionWithPromptCache: %v", err)
	}
	defer func() {
		if err := s.Close(); err != nil {
			tb.Fatalf("Close: %v", err)
		}
	}()
	if _, err := s.PrefillChunk(prompt); err != nil {
		tb.Fatalf("PrefillChunk(%v): %v", prompt, err)
	}
	for {
		finished, _ := s.Finished()
		if finished {
			break
		}
		if _, err := s.DecodeStep(); err != nil {
			tb.Fatalf("DecodeStep(%v): %v", prompt, err)
		}
	}
	return s.OutputTokens()
}

func mustWarmGemma4PromptCache(tb testing.TB, m *LlamaModel, cacheCfg Gemma4PromptCacheConfig, prompt []int) {
	tb.Helper()
	_ = mustRunGemma4PromptCacheSession(tb, m, cacheCfg, prompt, 0)
}

func TestGemma4PromptCacheSharedPrefixDivergentSuffixExactOutput(t *testing.T) {
	m := newGemma4PromptCacheTestModel()
	basePrompt, divergentPrompt, _ := gemma4PromptCacheSyntheticPrompts()
	wantBase := mustRunGemma4PromptCacheSession(t, m, Gemma4PromptCacheConfig{}, basePrompt, 3)
	wantDivergent := mustRunGemma4PromptCacheSession(t, m, Gemma4PromptCacheConfig{}, divergentPrompt, 3)

	for _, blockSize := range []int{2, 4} {
		t.Run(fmt.Sprintf("block_%d", blockSize), func(t *testing.T) {
			cache := promptcache.New(1 << 20)
			cfg := Gemma4PromptCacheConfig{Cache: cache, Identity: gemma4TestCacheIdentity(fmt.Sprintf("shared-prefix-%d", blockSize)), BlockSize: blockSize}

			gotBaseCold := mustRunGemma4PromptCacheSession(t, m, cfg, basePrompt, 3)
			gotDivergentPartial := mustRunGemma4PromptCacheSession(t, m, cfg, divergentPrompt, 3)
			gotBaseFull := mustRunGemma4PromptCacheSession(t, m, cfg, basePrompt, 3)

			if !sameInts(gotBaseCold, wantBase) {
				t.Fatalf("cold shared-prefix output=%v want %v", gotBaseCold, wantBase)
			}
			if !sameInts(gotDivergentPartial, wantDivergent) {
				t.Fatalf("partial divergent output=%v want %v", gotDivergentPartial, wantDivergent)
			}
			if !sameInts(gotBaseFull, wantBase) {
				t.Fatalf("full-hit repeated output=%v want %v", gotBaseFull, wantBase)
			}

			st := cache.Stats()
			if st.Hits < 2 {
				t.Fatalf("cache hits=%d want at least 2 after cold/partial/full sequence", st.Hits)
			}
			if st.UsedBytes == 0 || st.Entries == 0 {
				t.Fatalf("cache stats=%+v", st)
			}
		})
	}
}

func TestGemma4PromptCacheTinyBudgetEviction(t *testing.T) {
	m := newGemma4PromptCacheTestModel()
	promptA, _, promptB := gemma4PromptCacheSyntheticPrompts()
	const blockSize = 2
	largeCacheCfg := Gemma4PromptCacheConfig{Cache: promptcache.New(1 << 20), Identity: gemma4TestCacheIdentity("budget-size"), BlockSize: blockSize}
	mustWarmGemma4PromptCache(t, m, largeCacheCfg, promptA)
	budget := largeCacheCfg.Cache.Stats().UsedBytes
	if budget <= 0 {
		t.Fatalf("measured budget=%d", budget)
	}

	cache := promptcache.New(budget)
	cfg := Gemma4PromptCacheConfig{Cache: cache, Identity: gemma4TestCacheIdentity("tiny-budget"), BlockSize: blockSize}
	mustWarmGemma4PromptCache(t, m, cfg, promptA)
	mustWarmGemma4PromptCache(t, m, cfg, promptB)

	preparedA := m.PreparedGenerateTokens(promptA)
	preparedB := m.PreparedGenerateTokens(promptB)
	if snap, ok, err := cache.FindLongest(cfg.Identity, blockSize, preparedA); err != nil {
		t.Fatalf("FindLongest(promptA): %v", err)
	} else if ok || snap != nil {
		t.Fatalf("promptA should be evicted, ok=%v snap=%T", ok, snap)
	}
	if snap, ok, err := cache.FindLongest(cfg.Identity, blockSize, preparedB); err != nil {
		t.Fatalf("FindLongest(promptB): %v", err)
	} else if !ok || snap == nil {
		t.Fatalf("promptB should remain resident, ok=%v snap=%T", ok, snap)
	}

	st := cache.Stats()
	if st.Evictions == 0 {
		t.Fatalf("expected eviction, stats=%+v", st)
	}
	if st.UsedBytes > budget {
		t.Fatalf("used bytes=%d exceeded budget=%d", st.UsedBytes, budget)
	}
}

func TestGemma4PromptCacheConcurrentCloneIsolation(t *testing.T) {
	m := newGemma4PromptCacheTestModel()
	prompt, _, _ := gemma4PromptCacheSyntheticPrompts()
	cache := promptcache.New(1 << 20)
	cfg := Gemma4PromptCacheConfig{Cache: cache, Identity: gemma4TestCacheIdentity("clone-isolation"), BlockSize: 2}
	want := mustRunGemma4PromptCacheSession(t, m, Gemma4PromptCacheConfig{}, prompt, 2)
	mustWarmGemma4PromptCache(t, m, cfg, prompt)

	var wg sync.WaitGroup
	errCh := make(chan error, 16)
	for g := 0; g < 16; g++ {
		g := g
		wg.Add(1)
		go func() {
			defer wg.Done()
			s, err := NewGemma4DecodeSessionWithPromptCache(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 1}, cfg)
			if err != nil {
				errCh <- fmt.Errorf("session %d create: %w", g, err)
				return
			}
			defer s.Close()
			if _, err := s.PrefillChunk(prompt); err != nil {
				errCh <- fmt.Errorf("session %d prefill: %w", g, err)
				return
			}
			if len(s.pendingLogits) == 0 {
				errCh <- fmt.Errorf("session %d missing pending logits after cached prefill", g)
				return
			}
			if len(s.state.kvCacheK) == 0 || len(s.state.kvCacheV) == 0 || len(s.state.kvCacheK[0]) == 0 || len(s.state.kvCacheV[0]) == 0 {
				errCh <- fmt.Errorf("session %d missing restored KV", g)
				return
			}
			s.state.kvCacheK[0][0] = float32(100 + g)
			s.state.kvCacheV[0][0] = float32(200 + g)
			s.pendingLogits[0] = float32(300 + g)
			if _, err := s.DecodeStep(); err != nil {
				errCh <- fmt.Errorf("session %d decode: %w", g, err)
				return
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		t.Error(err)
	}
	if t.Failed() {
		return
	}

	got := mustRunGemma4PromptCacheSession(t, m, cfg, prompt, 2)
	if !sameInts(got, want) {
		t.Fatalf("post-race cached output=%v want %v", got, want)
	}
}
