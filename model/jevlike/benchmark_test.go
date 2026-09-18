package jevlike

import "testing"

// Benchmarks include tokenization, allocations, projections and probability
// calculation rather than timing an isolated dot product.
func BenchmarkTinyScoring(b *testing.B) {
	m, err := NewInitializedTinyScorer(Config{Width: 64, Rank: 64, ContextTokens: 192, OptionTokens: 32}, 7)
	if err != nil {
		b.Fatal(err)
	}
	examples := make([]ChoiceExample, 8)
	for i := range examples {
		examples[i] = SyntheticExample(int64(i + 1))
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		batch, err := BuildByteBatch(examples, 192, 32)
		if err != nil {
			b.Fatal(err)
		}
		if _, err = m.PredictBatch(batch); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkTinyTrainingEpoch(b *testing.B) {
	examples := make([]ChoiceExample, 32)
	for i := range examples {
		examples[i] = SyntheticExample(int64(i + 1))
	}
	cfg := DefaultTrainConfig()
	cfg.Epochs = 1
	cfg.BatchSize = 16
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m, err := NewInitializedTinyScorer(Config{Width: 64, Rank: 64, ContextTokens: 192, OptionTokens: 32}, 7)
		if err != nil {
			b.Fatal(err)
		}
		if _, err = TrainTinyScorer(m, examples, examples, cfg); err != nil {
			b.Fatal(err)
		}
	}
}
