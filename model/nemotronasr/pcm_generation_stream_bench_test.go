package nemotronasr

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Measures one complete 11-second PCM-to-raw-token request, excluding model
// and WAV loading. Each call owns bounded caches; output is consumed per call.
func BenchmarkPCMGenerationStreamJFK(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	model, err := LoadPCMGenerationModel(file)
	if err != nil {
		b.Fatal(err)
	}
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) != 176000 {
		b.Fatalf("JFK rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stream := &PCMGenerationStream{Model: model}
		var decisions int
		for start := 0; start < len(pcm); start += 80000 {
			end := start + 80000
			if end > len(pcm) {
				end = len(pcm)
			}
			tokens, _, err := stream.AppendPCM(ctx, pcm[start:end])
			if err != nil {
				b.Fatal(err)
			}
			decisions += len(tokens)
		}
		last, _, err := stream.Finish(ctx)
		if err != nil {
			b.Fatal(err)
		}
		if decisions+len(last) != 185 {
			b.Fatalf("JFK decisions=%d", decisions+len(last))
		}
	}
}
