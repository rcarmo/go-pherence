package model

import (
	"os"
	"testing"
)

func TestGemma4DecodeSessionUpdatedRealGGUFOneStep(t *testing.T) {
	if os.Getenv("GO_PHERENCE_GEMMA4_SESSION_REAL") == "" {
		t.Skip("set GO_PHERENCE_GEMMA4_SESSION_REAL=1 for the local updated Gemma4 session gate")
	}
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = "../models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf"
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("Gemma4 GGUF unavailable: %v", err)
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	// Token 10979 is the established local Gemma4 parity prompt token. The
	// session applies model-specific preparation exactly once during prefill.
	s, err := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	prefill, err := s.PrefillChunk([]int{10979})
	if err != nil {
		t.Fatal(err)
	}
	if !prefill.ReadyToDecode || prefill.Position <= 1 {
		t.Fatalf("unexpected prepared prefill=%+v", prefill)
	}
	step, err := s.DecodeStep()
	if err != nil {
		t.Fatal(err)
	}
	if step.Token < 0 || step.Token >= m.Config.VocabSize || len(step.Logits) != m.Config.VocabSize || !step.Finished || step.FinishReason != FinishReasonLength {
		t.Fatalf("unexpected decode step token=%d logits=%d result=%+v", step.Token, len(step.Logits), step)
	}
}
