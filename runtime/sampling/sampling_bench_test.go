package sampling

import "testing"

func benchmarkLogits(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*7919)%10007)/1000 - float32(i%13)
	}
	return out
}

func benchmarkSample(b *testing.B, vocab int, cfg Config) {
	logits := benchmarkLogits(vocab)
	b.ReportAllocs()
	b.SetBytes(int64(vocab * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := SampleWithDraw(logits, cfg, 0.371); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkGreedy32K(b *testing.B) { benchmarkSample(b, 32<<10, Config{}) }
func BenchmarkTopK32K(b *testing.B) {
	benchmarkSample(b, 32<<10, Config{Temperature: 0.8, TopK: 40})
}
func BenchmarkTopP32K(b *testing.B) {
	benchmarkSample(b, 32<<10, Config{Temperature: 0.8, TopP: 0.9})
}
func BenchmarkGreedy128K(b *testing.B) { benchmarkSample(b, 128<<10, Config{}) }
func BenchmarkTopK128K(b *testing.B) {
	benchmarkSample(b, 128<<10, Config{Temperature: 0.8, TopK: 40})
}
func BenchmarkTopP128K(b *testing.B) {
	benchmarkSample(b, 128<<10, Config{Temperature: 0.8, TopP: 0.9})
}
