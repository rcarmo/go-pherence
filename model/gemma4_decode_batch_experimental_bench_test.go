package model

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkGemma4ExperimentalDecodeBatchRealE4B(b *testing.B) {
	if os.Getenv("GO_PHERENCE_GEMMA4_BATCH_REAL") != "1" {
		b.Skip("set GO_PHERENCE_GEMMA4_BATCH_REAL=1")
	}
	path := os.Getenv("GO_PHERENCE_GEMMA4_MAIN")
	if path == "" {
		path = filepath.Join(findMTPGraphBenchRepoRoot(), "models", "gemma4-e4b-it-google-qat-gguf", "gemma-4-E4B_q4_0-it.gguf")
	}
	m, err := LoadGemma4GGUFAsLlama(path)
	if err != nil {
		b.Fatal(err)
	}
	for _, batch := range []int{1, 2, 4, 8} {
		for _, mode := range []string{"sequential", "batched_tail"} {
			b.Run(fmt.Sprintf("batch_%d/%s", batch, mode), func(b *testing.B) {
				for n := 0; n < b.N; n++ {
					sessions := make([]*Gemma4DecodeSession, batch)
					b.StopTimer()
					for i := range sessions {
						s, err := NewGemma4DecodeSession(m, SessionOptions{Backend: InferenceBackendSIMD, MaxTokens: 3})
						if err != nil {
							b.Fatal(err)
						}
						if _, err := s.PrefillChunk([]int{1}); err != nil {
							b.Fatal(err)
						}
						if _, err := s.DecodeStep(); err != nil { // consume boundary logits
							b.Fatal(err)
						}
						sessions[i] = s
					}
					b.StartTimer()
					if mode == "batched_tail" {
						if _, err := RunGemma4ExperimentalDecodeBatch(sessions); err != nil {
							b.Fatal(err)
						}
					} else {
						for _, s := range sessions {
							if _, err := s.DecodeStep(); err != nil {
								b.Fatal(err)
							}
						}
					}
				}
				b.ReportMetric(float64(batch), "tokens/op")
			})
		}
	}
}

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
