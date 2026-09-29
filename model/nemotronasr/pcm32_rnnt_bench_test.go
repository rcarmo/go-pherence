package nemotronasr

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Measures the owned fixed-prefix PCM-to-token call, excluding model and WAV loading.
func BenchmarkPCM32RNNTJFKPrefix(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	model, err := LoadPCM32RNNT(file)
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
	prefix := pcm[:5160]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := model.Decode(prefix, 3, 101); err != nil {
			b.Fatal(err)
		}
	}
}
