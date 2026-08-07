package model

import (
	"testing"

	"github.com/rcarmo/go-pherence/runtime/promptcache"
)

func gemma4TestCacheIdentity(salt string) promptcache.Identity {
	return promptcache.Identity{ModelFingerprint: "gemma4", CheckpointFingerprint: "test", Backend: "simd", WeightLayout: "test", WeightDType: "f32", KVPolicy: "linear", KVPrecision: "f32", ConfigFingerprint: "cfg", RoPEFingerprint: "rope", CacheSalt: salt}
}

func TestGemma4PromptSnapshotCloneAndSize(t *testing.T) {
	s := &Gemma4PromptSnapshot{EndPos: 2, KVCacheK: [][]float32{{1, 2}, nil}, KVCacheV: [][]float32{{3, 4}, nil}, BoundaryLogits: []float32{5, 6}, BoundaryToken: 1}
	n, err := s.SizeBytes()
	if err != nil || n <= 0 {
		t.Fatalf("size=%d err=%v", n, err)
	}
	clone := s.Clone().(*Gemma4PromptSnapshot)
	clone.KVCacheK[0][0] = 99
	clone.BoundaryLogits[0] = 99
	if s.KVCacheK[0][0] != 1 || s.BoundaryLogits[0] != 5 {
		t.Fatal("clone aliases source")
	}
}

func TestGemma4SessionPromptCacheFullAndPartialHits(t *testing.T) {
	m := newZeroLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	cache := promptcache.New(1 << 20)
	cfg := Gemma4PromptCacheConfig{Cache: cache, Identity: gemma4TestCacheIdentity("a"), BlockSize: 2}

	cold, err := NewGemma4DecodeSessionWithPromptCache(m, SessionOptions{MaxTokens: 2}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := cold.PrefillChunk([]int{1, 2}); err != nil {
		t.Fatal(err)
	}
	cold1, _ := cold.DecodeStep()
	cold2, _ := cold.DecodeStep()
	if cache.Stats().Entries == 0 {
		t.Fatal("cold prefill stored no prefix")
	}

	full, _ := NewGemma4DecodeSessionWithPromptCache(m, SessionOptions{MaxTokens: 2}, cfg)
	if _, err := full.PrefillChunk([]int{1, 2}); err != nil {
		t.Fatal(err)
	}
	full1, _ := full.DecodeStep()
	full2, _ := full.DecodeStep()
	if full1.Token != cold1.Token || full2.Token != cold2.Token {
		t.Fatalf("full hit tokens=%d,%d cold=%d,%d", full1.Token, full2.Token, cold1.Token, cold2.Token)
	}

	partial, _ := NewGemma4DecodeSessionWithPromptCache(m, SessionOptions{MaxTokens: 1}, cfg)
	if _, err := partial.PrefillChunk([]int{1, 2, 1}); err != nil {
		t.Fatal(err)
	}
	partialStep, _ := partial.DecodeStep()
	plain, _ := NewGemma4DecodeSession(m, SessionOptions{MaxTokens: 1})
	if _, err := plain.PrefillChunk([]int{1, 2, 1}); err != nil {
		t.Fatal(err)
	}
	plainStep, _ := plain.DecodeStep()
	if partialStep.Token != plainStep.Token {
		t.Fatalf("partial hit token=%d cold=%d", partialStep.Token, plainStep.Token)
	}
	if st := cache.Stats(); st.Hits < 2 {
		t.Fatalf("cache hits=%d want at least 2", st.Hits)
	}
}

func TestGemma4PromptCacheConfigValidation(t *testing.T) {
	m := newZeroLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	if _, err := NewGemma4DecodeSessionWithPromptCache(m, SessionOptions{MaxTokens: 1}, Gemma4PromptCacheConfig{Cache: promptcache.New(1), BlockSize: 2}); err == nil {
		t.Fatal("accepted missing identity")
	}
	if _, err := NewGemma4DecodeSessionWithPromptCache(m, SessionOptions{MaxTokens: 1}, Gemma4PromptCacheConfig{Cache: promptcache.New(1), Identity: gemma4TestCacheIdentity("a")}); err == nil {
		t.Fatal("accepted zero block size")
	}
}
