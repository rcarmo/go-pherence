package model

import "testing"

func TestGemma4DecodeSessionValidation(t *testing.T) {
	if _, err := NewGemma4DecodeSession(nil, SessionOptions{}); err == nil {
		t.Fatal("accepted nil model")
	}
	m := newZeroLayerVerifierModel()
	if _, err := NewGemma4DecodeSession(m, SessionOptions{}); err == nil {
		t.Fatal("accepted non-Gemma4 model")
	}
	m.Config.ModelType = "gemma4_text"
	if _, err := NewGemma4DecodeSession(m, SessionOptions{MaxTokens: -1}); err == nil {
		t.Fatal("accepted negative max tokens")
	}
	for _, backend := range []InferenceBackend{InferenceBackendScalar, InferenceBackendNVIDIA} {
		if _, err := NewGemma4DecodeSession(m, SessionOptions{Backend: backend}); err == nil {
			t.Fatalf("accepted unimplemented %s session", backend)
		}
	}
}

func TestGemma4DecodeSessionLifecycleAndCheckpoint(t *testing.T) {
	m := newZeroLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	s, err := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 3})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.DecodeStep(); err == nil {
		t.Fatal("decoded before prefill")
	}
	prefill, err := s.PrefillChunk([]int{1})
	if err != nil || prefill.ConsumedTokens != 1 || !prefill.ReadyToDecode {
		t.Fatalf("prefill=%+v err=%v", prefill, err)
	}
	cp, err := s.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	first, err := s.DecodeStep()
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Restore(cp); err != nil {
		t.Fatal(err)
	}
	again, err := s.DecodeStep()
	if err != nil {
		t.Fatal(err)
	}
	if first.Token != again.Token || len(first.Logits) != len(again.Logits) {
		t.Fatalf("restored step first=%+v again=%+v", first, again)
	}
	for !again.Finished {
		again, err = s.DecodeStep()
		if err != nil {
			t.Fatal(err)
		}
	}
	if again.Generated != 3 || again.FinishReason != FinishReasonLength {
		t.Fatalf("final=%+v", again)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.DecodeStep(); err == nil {
		t.Fatal("decoded after close")
	}
}

func TestGemma4DecodeSessionKVStatsTrackAndRestoreCapacity(t *testing.T) {
	m := newGemma4SingleLayerDecodeSessionTestModel()
	s, err := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 3})
	if err != nil {
		t.Fatal(err)
	}
	before, err := s.KVStats()
	if err != nil || before.UsedBytes != 0 || before.ReservedBytes != 0 {
		t.Fatalf("before=%+v err=%v", before, err)
	}
	if _, err := s.PrefillChunk([]int{1}); err != nil {
		t.Fatal(err)
	}
	prefilled, err := s.KVStats()
	if err != nil {
		t.Fatal(err)
	}
	if len(prefilled.Layers) != 1 || prefilled.UsedBytes <= 0 || prefilled.ReservedBytes < prefilled.UsedBytes || prefilled.UnusedReservedBytes != prefilled.ReservedBytes-prefilled.UsedBytes {
		t.Fatalf("prefilled=%+v", prefilled)
	}
	cp, err := s.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.DecodeStep(); err != nil {
		t.Fatal(err)
	}
	staged, err := s.KVStats()
	if err != nil || staged.UsedBytes < prefilled.UsedBytes || staged.ReservedBytes < staged.UsedBytes {
		t.Fatalf("staged=%+v err=%v", staged, err)
	}
	if err := s.Restore(cp); err != nil {
		t.Fatal(err)
	}
	restored, err := s.KVStats()
	if err != nil || restored.UsedBytes != prefilled.UsedBytes || restored.ReservedBytes < restored.UsedBytes {
		t.Fatalf("restored=%+v prefilled=%+v err=%v", restored, prefilled, err)
	}
}

func TestGemma4DecodeSessionRejectsForeignCheckpoint(t *testing.T) {
	m := newZeroLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	a, _ := NewGemma4DecodeSession(m, SessionOptions{MaxTokens: 1})
	b, _ := NewGemma4DecodeSession(m, SessionOptions{MaxTokens: 1})
	_, _ = a.PrefillChunk([]int{1})
	_, _ = b.PrefillChunk([]int{1})
	cp, err := a.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Restore(cp); err == nil {
		t.Fatal("accepted foreign checkpoint")
	}
}

func TestInferenceSessionOptionErrors(t *testing.T) {
	if err := validateSessionOptions(SessionOptions{Backend: "other"}); err == nil {
		t.Fatal("accepted unsupported backend")
	}
	if err := validateSessionOptions(SessionOptions{StopTokenIDs: []int{-1}}); err == nil {
		t.Fatal("accepted negative stop token")
	}
}
