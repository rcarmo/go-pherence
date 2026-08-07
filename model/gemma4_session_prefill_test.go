package model

import (
	"fmt"
	"testing"

	"github.com/rcarmo/go-pherence/runtime/promptcache"
)

type gemma4SessionTrace struct {
	prefillPosition int
	prefillKVCacheK [][]float32
	prefillKVCacheV [][]float32
	decode          []DecodeResult
	output          []int
}

func newGemma4SingleLayerDecodeSessionTestModel() *LlamaModel {
	m := newSingleLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	return m
}

func newGemma4DecodeSessionForTest(t *testing.T, m *LlamaModel, maxTokens int, cacheCfg Gemma4PromptCacheConfig) *Gemma4DecodeSession {
	t.Helper()
	var (
		s   *Gemma4DecodeSession
		err error
	)
	if cacheCfg.enabled() {
		s, err = NewGemma4DecodeSessionWithPromptCache(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: maxTokens}, cacheCfg)
	} else {
		s, err = NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: maxTokens})
	}
	if err != nil {
		t.Fatalf("new Gemma4 decode session: %v", err)
	}
	return s
}

func collectGemma4SessionTrace(t *testing.T, s *Gemma4DecodeSession) gemma4SessionTrace {
	t.Helper()
	if s == nil || !s.prefilled || s.state == nil {
		t.Fatal("Gemma4 session is not ready to trace")
	}
	trace := gemma4SessionTrace{
		prefillPosition: s.state.position,
		prefillKVCacheK: cloneFloat32Matrix(s.state.kvCacheK),
		prefillKVCacheV: cloneFloat32Matrix(s.state.kvCacheV),
	}
	for {
		step, err := s.DecodeStep()
		if err != nil {
			t.Fatalf("DecodeStep: %v", err)
		}
		trace.decode = append(trace.decode, step)
		if step.Finished {
			break
		}
	}
	trace.output = s.OutputTokens()
	return trace
}

func runGemma4SessionOneShotTrace(t *testing.T, m *LlamaModel, prompt []int, maxTokens int, cacheCfg Gemma4PromptCacheConfig) gemma4SessionTrace {
	t.Helper()
	s := newGemma4DecodeSessionForTest(t, m, maxTokens, cacheCfg)
	prefill, err := s.PrefillChunk(prompt)
	if err != nil {
		t.Fatalf("PrefillChunk: %v", err)
	}
	prepared := m.PreparedGenerateTokens(prompt)
	if prefill.ConsumedTokens != len(prepared) || prefill.Position != len(prepared) || !prefill.ReadyToDecode {
		t.Fatalf("prefill=%+v prepared=%d", prefill, len(prepared))
	}
	return collectGemma4SessionTrace(t, s)
}

func assertGemma4SessionTraceEqual(t *testing.T, got, want gemma4SessionTrace) {
	t.Helper()
	if got.prefillPosition != want.prefillPosition {
		t.Fatalf("prefill position=%d want %d", got.prefillPosition, want.prefillPosition)
	}
	for name, pair := range map[string][2][][]float32{
		"KV K": {got.prefillKVCacheK, want.prefillKVCacheK},
		"KV V": {got.prefillKVCacheV, want.prefillKVCacheV},
	} {
		if len(pair[0]) != len(pair[1]) {
			t.Fatalf("%s layers=%d want %d", name, len(pair[0]), len(pair[1]))
		}
		for i := range pair[0] {
			if !sameFloat32s(pair[0][i], pair[1][i]) {
				t.Fatalf("%s layer %d=%v want %v", name, i, pair[0][i], pair[1][i])
			}
		}
	}
	if !sameInts(got.output, want.output) {
		t.Fatalf("output=%v want %v", got.output, want.output)
	}
	if len(got.decode) != len(want.decode) {
		t.Fatalf("decode steps=%d want %d", len(got.decode), len(want.decode))
	}
	for i := range got.decode {
		g, w := got.decode[i], want.decode[i]
		if g.Token != w.Token || g.Position != w.Position || g.Generated != w.Generated || g.Finished != w.Finished || g.FinishReason != w.FinishReason {
			t.Fatalf("decode step %d=%+v want %+v", i, g, w)
		}
		if !sameFloat32s(g.Logits, w.Logits) {
			t.Fatalf("decode logits[%d]=%v want %v", i, g.Logits, w.Logits)
		}
	}
}

func TestGemma4DecodeSessionBoundedPrefillMatchesOneShot(t *testing.T) {
	m := newGemma4SingleLayerDecodeSessionTestModel()
	prompt := []int{0, 1, 2, 1, 0}
	prepared := m.PreparedGenerateTokens(prompt)
	want := runGemma4SessionOneShotTrace(t, m, prompt, 3, Gemma4PromptCacheConfig{})
	cases := []struct {
		name         string
		chunkSize    int
		wantConsumed []int
	}{
		{name: "chunk1", chunkSize: 1, wantConsumed: []int{1, 1, 1, 1, 1}},
		{name: "chunk2", chunkSize: 2, wantConsumed: []int{2, 2, 1}},
		{name: "chunk3", chunkSize: 3, wantConsumed: []int{3, 2}},
		{name: "full", chunkSize: len(prepared), wantConsumed: []int{len(prepared)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := newGemma4DecodeSessionForTest(t, m, 3, Gemma4PromptCacheConfig{})
			if err := s.BeginPrefill(prompt); err != nil {
				t.Fatalf("BeginPrefill: %v", err)
			}
			if got := s.RemainingPrefill(); got != len(prepared) {
				t.Fatalf("initial remaining=%d want %d", got, len(prepared))
			}
			pos := 0
			for i, wantConsumed := range tc.wantConsumed {
				res, err := s.PrefillNext(tc.chunkSize)
				if err != nil {
					t.Fatalf("PrefillNext(%d): %v", tc.chunkSize, err)
				}
				if res.ConsumedTokens != wantConsumed {
					t.Fatalf("chunk %d consumed=%d want %d", i, res.ConsumedTokens, wantConsumed)
				}
				pos += wantConsumed
				if res.Position != pos {
					t.Fatalf("chunk %d position=%d want %d", i, res.Position, pos)
				}
				if res.ReadyToDecode != (i == len(tc.wantConsumed)-1) {
					t.Fatalf("chunk %d ready=%v", i, res.ReadyToDecode)
				}
				if got := s.RemainingPrefill(); got != len(prepared)-pos {
					t.Fatalf("chunk %d remaining=%d want %d", i, got, len(prepared)-pos)
				}
			}
			got := collectGemma4SessionTrace(t, s)
			assertGemma4SessionTraceEqual(t, got, want)
		})
	}
}

func TestGemma4DecodeSessionBoundedPrefillValidation(t *testing.T) {
	m := newGemma4SingleLayerDecodeSessionTestModel()
	s := newGemma4DecodeSessionForTest(t, m, 2, Gemma4PromptCacheConfig{})
	if _, err := s.PrefillNext(1); err == nil {
		t.Fatal("accepted PrefillNext before BeginPrefill")
	}
	if err := s.BeginPrefill([]int{0, 1, 2}); err != nil {
		t.Fatalf("BeginPrefill: %v", err)
	}
	if err := s.BeginPrefill([]int{0, 1, 2}); err == nil {
		t.Fatal("accepted repeated BeginPrefill")
	}
	if _, err := s.DecodeStep(); err == nil {
		t.Fatal("decoded before prefill completed")
	}
	for _, limit := range []int{0, -1} {
		if _, err := s.PrefillNext(limit); err == nil {
			t.Fatalf("accepted PrefillNext(%d)", limit)
		}
	}
	res, err := s.PrefillNext(2)
	if err != nil {
		t.Fatalf("PrefillNext(2): %v", err)
	}
	if res.ConsumedTokens != 2 || res.Position != 2 || res.ReadyToDecode {
		t.Fatalf("first chunk=%+v", res)
	}
	if got := s.RemainingPrefill(); got != 1 {
		t.Fatalf("remaining=%d want 1", got)
	}
	if _, err := s.DecodeStep(); err == nil {
		t.Fatal("decoded before final prefill tail completed")
	}
	res, err = s.PrefillNext(2)
	if err != nil {
		t.Fatalf("PrefillNext tail: %v", err)
	}
	if res.ConsumedTokens != 1 || res.Position != 3 || !res.ReadyToDecode {
		t.Fatalf("tail chunk=%+v", res)
	}
	if got := s.RemainingPrefill(); got != 0 {
		t.Fatalf("remaining=%d want 0", got)
	}
	if _, err := s.DecodeStep(); err != nil {
		t.Fatalf("DecodeStep after prefill completion: %v", err)
	}
}

func TestGemma4DecodeSessionBoundedPrefillPromptCacheRestore(t *testing.T) {
	m := newGemma4SingleLayerDecodeSessionTestModel()
	cache := promptcache.New(1 << 20)
	cfg := Gemma4PromptCacheConfig{Cache: cache, Identity: gemma4TestCacheIdentity("bounded-prefill"), BlockSize: 2}
	warmPrompt := []int{0, 1, 2, 1}
	warm := newGemma4DecodeSessionForTest(t, m, 2, cfg)
	if _, err := warm.PrefillChunk(warmPrompt); err != nil {
		t.Fatalf("warm PrefillChunk: %v", err)
	}
	if cache.Stats().Entries == 0 {
		t.Fatal("warm prefill stored no prompt cache entries")
	}

	t.Run("full_restore", func(t *testing.T) {
		want := runGemma4SessionOneShotTrace(t, m, warmPrompt, 2, Gemma4PromptCacheConfig{})
		before := cache.Stats()
		s := newGemma4DecodeSessionForTest(t, m, 2, cfg)
		if err := s.BeginPrefill(warmPrompt); err != nil {
			t.Fatalf("BeginPrefill: %v", err)
		}
		afterBegin := cache.Stats()
		if got := s.RemainingPrefill(); got != 0 {
			t.Fatalf("remaining=%d want 0", got)
		}
		prepared := m.PreparedGenerateTokens(warmPrompt)
		if s.state.position != len(prepared) {
			t.Fatalf("restored position=%d want %d", s.state.position, len(prepared))
		}
		if afterBegin.Lookups != before.Lookups+1 || afterBegin.Hits != before.Hits+1 {
			t.Fatalf("cache stats after begin=%+v before=%+v", afterBegin, before)
		}
		got := collectGemma4SessionTrace(t, s)
		assertGemma4SessionTraceEqual(t, got, want)
		afterTrace := cache.Stats()
		if afterTrace.Lookups != afterBegin.Lookups || afterTrace.Hits != afterBegin.Hits {
			t.Fatalf("cache lookup/hit changed after full restore trace begin=%+v after=%+v", afterBegin, afterTrace)
		}
	})

	t.Run("partial_restore", func(t *testing.T) {
		partialPrompt := []int{0, 1, 2, 1, 0}
		want := runGemma4SessionOneShotTrace(t, m, partialPrompt, 2, Gemma4PromptCacheConfig{})
		before := cache.Stats()
		s := newGemma4DecodeSessionForTest(t, m, 2, cfg)
		if err := s.BeginPrefill(partialPrompt); err != nil {
			t.Fatalf("BeginPrefill: %v", err)
		}
		afterBegin := cache.Stats()
		if got := s.RemainingPrefill(); got != 1 {
			t.Fatalf("remaining=%d want 1", got)
		}
		preparedWarm := m.PreparedGenerateTokens(warmPrompt)
		if s.state.position != len(preparedWarm) {
			t.Fatalf("restored position=%d want %d", s.state.position, len(preparedWarm))
		}
		if afterBegin.Lookups != before.Lookups+1 || afterBegin.Hits != before.Hits+1 {
			t.Fatalf("cache stats after begin=%+v before=%+v", afterBegin, before)
		}
		if _, err := s.DecodeStep(); err == nil {
			t.Fatal("decoded before restored tail prefill completed")
		}
		res, err := s.PrefillNext(3)
		if err != nil {
			t.Fatalf("PrefillNext: %v", err)
		}
		prepared := m.PreparedGenerateTokens(partialPrompt)
		if res.ConsumedTokens != 1 || res.Position != len(prepared) || !res.ReadyToDecode {
			t.Fatalf("tail prefill=%+v prepared=%d", res, len(prepared))
		}
		afterPrefill := cache.Stats()
		if afterPrefill.Lookups != afterBegin.Lookups || afterPrefill.Hits != afterBegin.Hits {
			t.Fatalf("cache lookup/hit changed during PrefillNext begin=%+v after=%+v", afterBegin, afterPrefill)
		}
		got := collectGemma4SessionTrace(t, s)
		assertGemma4SessionTraceEqual(t, got, want)
	})
}

func (t gemma4SessionTrace) String() string {
	return fmt.Sprintf("prefill=%d decode=%d output=%v", t.prefillPosition, len(t.decode), t.output)
}
