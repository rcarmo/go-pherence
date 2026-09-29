package nemotronasr

import (
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// The pinned full-JFK projected input excludes the encoder and model loading.
func BenchmarkGreedyRNNTJFKFullProjected(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	decoder, err := LoadRNNTDecoder(file)
	if err != nil {
		b.Fatal(err)
	}
	projection, err := LoadRNNTProjection(file)
	if err != nil {
		b.Fatal(err)
	}
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	encoder := readStemFixture(b, "asr_jfk_full_encoder", 139*rnntHidden)
	greedy := &GreedyRNNT{Decoder: decoder, Projection: projection}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := greedy.Decode(encoder, 139); err != nil {
			b.Fatal(err)
		}
	}
}
