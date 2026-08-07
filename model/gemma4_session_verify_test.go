package model

import "testing"

func newGemma4VerifyTestSession(t *testing.T, maxTokens int, stops ...int) *Gemma4DecodeSession {
	t.Helper()
	m := newZeroLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	s, err := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: maxTokens, StopTokenIDs: stops})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.PrefillChunk([]int{1}); err != nil {
		t.Fatal(err)
	}
	return s
}

func decodeGemma4VerifyTokens(t *testing.T, s *Gemma4DecodeSession, n int) []int {
	t.Helper()
	out := make([]int, n)
	for i := range out {
		step, err := s.DecodeStep()
		if err != nil {
			t.Fatal(err)
		}
		out[i] = step.Token
	}
	return out
}

func TestGemma4DecodeSessionVerifyAllAcceptedAndBonusParity(t *testing.T) {
	oracle := newGemma4VerifyTestSession(t, 8)
	candidate := newGemma4VerifyTestSession(t, 8)
	want := decodeGemma4VerifyTokens(t, oracle, 3)
	got, err := candidate.Verify(want[:2])
	if err != nil {
		t.Fatal(err)
	}
	if !got.AllDraftsAccepted || got.AcceptedPrefixLen != 2 || !sameInts(got.OutputTokens, want) {
		t.Fatalf("acceptance=%+v want output=%v", got, want)
	}
	if output := candidate.OutputTokens(); !sameInts(output[len(output)-3:], want) {
		t.Fatalf("session output=%v want suffix=%v", output, want)
	}
	oracleNext := decodeGemma4VerifyTokens(t, oracle, 1)[0]
	candidateNext := decodeGemma4VerifyTokens(t, candidate, 1)[0]
	if candidateNext != oracleNext {
		t.Fatalf("post-verify token=%d want oracle=%d", candidateNext, oracleNext)
	}
}

func TestGemma4DecodeSessionVerifyRejectsDraftAndCommitsBonus(t *testing.T) {
	oracle := newGemma4VerifyTestSession(t, 8)
	candidate := newGemma4VerifyTestSession(t, 8)
	want := decodeGemma4VerifyTokens(t, oracle, 2)
	wrong := (want[0] + 1) % candidate.model.Config.VocabSize
	got, err := candidate.Verify([]int{wrong})
	if err != nil {
		t.Fatal(err)
	}
	if got.AllDraftsAccepted || got.AcceptedPrefixLen != 0 || got.FirstRejectedIndex != 0 || !sameInts(got.OutputTokens, want[:1]) {
		t.Fatalf("acceptance=%+v want bonus=%d", got, want[0])
	}
	if next := decodeGemma4VerifyTokens(t, candidate, 1)[0]; next != want[1] {
		t.Fatalf("post-rejection token=%d want=%d", next, want[1])
	}
}

func TestGemma4DecodeSessionVerifyRestoresOnCapacityAndStop(t *testing.T) {
	capacity := newGemma4VerifyTestSession(t, 2)
	before := capacity.OutputTokens()
	if _, err := capacity.Verify([]int{0, 0}); err == nil {
		t.Fatal("verification accepted insufficient generation capacity")
	}
	if after := capacity.OutputTokens(); !sameInts(after, before) {
		t.Fatalf("capacity error mutated output before=%v after=%v", before, after)
	}

	oracle := newGemma4VerifyTestSession(t, 4)
	stop := decodeGemma4VerifyTokens(t, oracle, 1)[0]
	stopped := newGemma4VerifyTestSession(t, 4, stop)
	before = stopped.OutputTokens()
	if _, err := stopped.Verify([]int{stop}); err == nil {
		t.Fatal("verification crossed a stop-token boundary")
	}
	if after := stopped.OutputTokens(); !sameInts(after, before) {
		t.Fatalf("stop error mutated output before=%v after=%v", before, after)
	}
	step, err := stopped.DecodeStep()
	if err != nil || step.Token != stop || !step.Finished || step.FinishReason != FinishReasonStopToken {
		t.Fatalf("ordinary stop step=%+v err=%v", step, err)
	}
}
