package gliner2

import (
	"os"
	"testing"
)

// Includes schema preparation, tokenizer, encoder, candidate generation and
// reranking. Weight loading is deliberately outside the timed region.
func BenchmarkPublishedEntityPipeline(b *testing.B) {
	dir := os.Getenv("GLINER_MODEL_DIR")
	if dir == "" {
		b.Skip("set GLINER_MODEL_DIR")
	}
	m, err := LoadEntityModel(dir)
	if err != nil {
		b.Fatal(err)
	}
	text := "Ada Lovelace lived in London."
	labels := []string{"person", "location"}
	if _, err = m.ScoreEntities(text, labels, 512); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		scores, err := m.ScoreEntities(text, labels, 512)
		if err != nil {
			b.Fatal(err)
		}
		if _, err = DecodeConfiguredEntities(text, scores, .5, "flat", m.Config.BoundaryHead); err != nil {
			b.Fatal(err)
		}
	}
}
