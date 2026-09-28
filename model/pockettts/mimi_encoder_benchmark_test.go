package pockettts

import (
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func releasedMimiEncoder(tb testing.TB) (*MimiEncoderCPU, []float32) {
	tb.Helper()
	path := os.Getenv("GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	if path == "" {
		tb.Skip("set GO_PHERENCE_POCKETTTS_VOICE_MODEL")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		tb.Fatal(err)
	}
	encoder, err := LoadMimiEncoderCPU(file, releasedConfig(tb))
	file.Close()
	if err != nil {
		tb.Fatal(err)
	}
	audio := make([]float32, 30*SampleRate)
	for i := range audio {
		audio[i] = .18*float32(math.Sin(2*math.Pi*220*float64(i)/SampleRate)) + .07*float32(math.Sin(2*math.Pi*731*float64(i)/SampleRate))
	}
	return encoder, audio
}

func BenchmarkReleasedMimiEncoderThirtySeconds(b *testing.B) {
	encoder, audio := releasedMimiEncoder(b)
	b.ReportAllocs()
	b.SetBytes(int64(len(audio) * 4))
	b.ResetTimer()
	for b.Loop() {
		latents, frames, err := encoder.Encode(audio)
		if err != nil {
			b.Fatal(err)
		}
		if frames != 375 || len(latents) != 375*32 {
			b.Fatalf("shape %d %d", frames, len(latents))
		}
	}
}
func BenchmarkReleasedMimiEncoderWorkspaceSetup(b *testing.B) {
	encoder, _ := releasedMimiEncoder(b)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := encoder.NewWorkspace(375); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkReleasedMimiEncoderThirtySecondsWarm(b *testing.B) {
	encoder, audio := releasedMimiEncoder(b)
	workspace, err := encoder.NewWorkspace(375)
	if err != nil {
		b.Fatal(err)
	}
	out := make([]float32, 375*32)
	if err = encoder.EncodeInto(out, audio, workspace); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(audio) * 4))
	b.ResetTimer()
	for b.Loop() {
		if err = encoder.EncodeInto(out, audio, workspace); err != nil {
			b.Fatal(err)
		}
	}
}
