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
	runGemma4SessionRealTrajectory(t, m, 1)
}

func TestGemma4DecodeSessionUpdatedRealGGUFTwoSteps(t *testing.T) {
	if os.Getenv("GO_PHERENCE_GEMMA4_SESSION_REAL_LONG") == "" {
		t.Skip("set GO_PHERENCE_GEMMA4_SESSION_REAL_LONG=1 for the two-step local updated Gemma4 session gate")
	}
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = "../models/gemma4-e4b-it-google-qat-gguf/gemma-4-E4B_q4_0-it.gguf"
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		t.Fatal(err)
	}
	runGemma4SessionRealTrajectory(t, m, 2)
}

func runGemma4SessionRealTrajectory(t *testing.T, m *LlamaModel, steps int) {
	t.Helper()
	// Token 10979 is the established local Gemma4 parity prompt token. The
	// session applies model-specific preparation exactly once during prefill.
	s, err := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: steps})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	prefill, err := s.PrefillChunk([]int{10979})
	if err != nil {
		t.Fatal(err)
	}
	if !prefill.ReadyToDecode || prefill.Position <= 1 || s.BootstrapReplay() {
		t.Fatalf("unexpected prepared prefill=%+v bootstrap_replay=%v", prefill, s.BootstrapReplay())
	}
	prepared := s.OutputTokens()
	want := []int{106, 236789}
	if steps > len(want) {
		t.Fatalf("fixture has %d generated tokens, requested %d", len(want), steps)
	}
	checkpoint, err := s.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	legacy := m.generatePrepared(prepared, steps)
	if len(legacy) != len(prepared)+steps {
		t.Fatalf("legacy output len=%d, want %d", len(legacy), len(prepared)+steps)
	}
	if !sameInts(legacy[len(prepared):], want[:steps]) {
		t.Fatalf("legacy generated=%v, frozen fixture=%v", legacy[len(prepared):], want[:steps])
	}
	for i := 0; i < steps; i++ {
		step, err := s.DecodeStep()
		if err != nil {
			t.Fatal(err)
		}
		if step.Token < 0 || step.Token >= m.Config.VocabSize || len(step.Logits) != m.Config.VocabSize {
			t.Fatalf("unexpected decode step %d token=%d logits=%d result=%+v", i, step.Token, len(step.Logits), step)
		}
		if step.Token != legacy[len(prepared)+i] || step.Token != want[i] {
			t.Fatalf("step %d session token=%d, legacy token=%d, frozen token=%d", i, step.Token, legacy[len(prepared)+i], want[i])
		}
		if step.Finished != (i == steps-1) {
			t.Fatalf("step %d finished=%v, want %v", i, step.Finished, i == steps-1)
		}
	}
	if err := s.Restore(checkpoint); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < steps; i++ {
		step, err := s.DecodeStep()
		if err != nil {
			t.Fatal(err)
		}
		if step.Token != want[i] {
			t.Fatalf("restored step %d token=%d, frozen token=%d", i, step.Token, want[i])
		}
	}
}
