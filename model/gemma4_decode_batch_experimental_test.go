package model

import (
	"slices"
	"testing"
)

func TestGemma4ExperimentalDecodeBatchMatchesIndependentSessions(t *testing.T) {
	for _, batch := range []int{1, 2, 4, 8} {
		t.Run(string(rune('0'+batch)), func(t *testing.T) {
			m := newZeroLayerVerifierModel()
			m.Config.ModelType = "gemma4_text"
			batched := make([]*Gemma4DecodeSession, batch)
			oracle := make([]*Gemma4DecodeSession, batch)
			for i := 0; i < batch; i++ {
				prompt := []int{1}
				if i&1 != 0 {
					prompt = []int{1, 2}
				}
				for _, dst := range [][]*Gemma4DecodeSession{batched, oracle} {
					s, err := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 3})
					if err != nil {
						t.Fatal(err)
					}
					if _, err := s.PrefillChunk(prompt); err != nil {
						t.Fatal(err)
					}
					dst[i] = s
				}
			}

			// Boundary logits are already available after prefill, so this first
			// step intentionally exercises the exact sequential fallback.
			first, err := RunGemma4ExperimentalDecodeBatch(batched)
			if err != nil {
				t.Fatal(err)
			}
			if first.UsedBatchFinish || first.FallbackReason == "" {
				t.Fatalf("boundary result=%+v", first)
			}
			for i := range oracle {
				want, err := oracle[i].DecodeStep()
				if err != nil {
					t.Fatal(err)
				}
				assertDecodeResultEqual(t, first.Results[i], want)
			}

			second, err := RunGemma4ExperimentalDecodeBatch(batched)
			if err != nil {
				t.Fatal(err)
			}
			if !second.UsedBatchFinish || second.FallbackReason != "" {
				t.Fatalf("tail result used=%v fallback=%q", second.UsedBatchFinish, second.FallbackReason)
			}
			if second.Metadata.Batch != batch || len(second.Metadata.TokensFlat) != batch || len(second.Metadata.PositionsFlat) != batch || len(second.Metadata.OutputLengthsFlat) != batch {
				t.Fatalf("metadata=%+v", second.Metadata)
			}
			for i := range oracle {
				want, err := oracle[i].DecodeStep()
				if err != nil {
					t.Fatal(err)
				}
				assertDecodeResultEqual(t, second.Results[i], want)
				if !slices.Equal(batched[i].OutputTokens(), oracle[i].OutputTokens()) {
					t.Fatalf("row %d output=%v want=%v", i, batched[i].OutputTokens(), oracle[i].OutputTokens())
				}
			}
		})
	}
}

func TestGemma4ExperimentalDecodeBatchValidation(t *testing.T) {
	if _, err := RunGemma4ExperimentalDecodeBatch(nil); err == nil {
		t.Fatal("accepted empty batch")
	}
	m := newZeroLayerVerifierModel()
	m.Config.ModelType = "gemma4_text"
	s, err := NewGemma4DecodeSession(m, SessionOptions{MaxTokens: 2})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := RunGemma4ExperimentalDecodeBatch([]*Gemma4DecodeSession{s}); err == nil {
		t.Fatal("accepted unprefilled session")
	}
	if _, err := RunGemma4ExperimentalDecodeBatch([]*Gemma4DecodeSession{s, s, s}); err == nil {
		t.Fatal("accepted unsupported batch size")
	}
}

func assertDecodeResultEqual(t *testing.T, got, want DecodeResult) {
	t.Helper()
	if got.Token != want.Token || got.Position != want.Position || got.Generated != want.Generated || got.Finished != want.Finished || got.FinishReason != want.FinishReason || !slices.Equal(got.Logits, want.Logits) {
		t.Fatalf("got=%+v want=%+v", got, want)
	}
}
