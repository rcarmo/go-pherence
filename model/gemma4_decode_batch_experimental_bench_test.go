package model

import (
	"fmt"
	"testing"
)

func BenchmarkGemma4ExperimentalDecodeBatch(b *testing.B) {
	for _, batch := range []int{1, 2, 4, 8} {
		b.Run(fmt.Sprintf("batch_%d", batch), func(b *testing.B) {
			m := newZeroLayerVerifierModel()
			m.Config.ModelType = "gemma4_text"
			sessions := make([]*Gemma4DecodeSession, batch)
			for i := range sessions {
				s, err := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: b.N + 2})
				if err != nil {
					b.Fatal(err)
				}
				if _, err := s.PrefillChunk([]int{1}); err != nil {
					b.Fatal(err)
				}
				// Consume prefill-boundary logits outside the measured batched tail.
				if _, err := s.DecodeStep(); err != nil {
					b.Fatal(err)
				}
				sessions[i] = s
			}
			b.ReportMetric(float64(batch), "requests/op")
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := RunGemma4ExperimentalDecodeBatch(sessions); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
