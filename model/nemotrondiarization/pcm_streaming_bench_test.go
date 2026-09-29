package nemotrondiarization

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// Bounded 30-second diagnostic, excluding checkpoint and WAV loading.
func BenchmarkPCMStreamingRequestThirtySeconds(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		b.Skip("set released model path")
	}
	f, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	prototype, err := LoadPCMStreamingRequest(f)
	if err != nil {
		b.Fatal(err)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 {
		b.Fatalf("JFK rate=%d err=%v", rate, err)
	}
	samples := make([]float32, 30*rate)
	for i := range samples {
		samples[i] = pcm[i%len(pcm)]
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache, err := NewSpeakerCache(prototype.window.Cache.compressor)
		if err != nil {
			b.Fatal(err)
		}
		s := &PCMStreamingRequest{frontend: &PCMStackingStream{stack: StackingStream{Projection: prototype.frontend.stack.Projection}}, window: &StreamingWindow{Tower: prototype.window.Tower, Head: prototype.window.Head, Cache: cache}}
		var count int
		for start := 0; start < len(samples); start += 80000 {
			end := min(start+80000, len(samples))
			out, err := s.AppendPCM(samples[start:end])
			if err != nil {
				b.Fatal(err)
			}
			count += len(out) / diarizationSpeakers
		}
		out, err := s.Finish()
		if err != nil {
			b.Fatal(err)
		}
		count += len(out) / diarizationSpeakers
		if count != 2999 {
			b.Fatalf("rows=%d", count)
		}
	}
	b.StopTimer()
	if err := f.Close(); err != nil {
		b.Fatal(err)
	}
}
