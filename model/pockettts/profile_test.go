package pockettts

import "testing"

func releasedWarmSession(tb testing.TB, frames int) (*Session, NoiseSource, []float32) {
	tb.Helper()
	modelPath := releasedModel(tb)
	voicePath := releasedVoice(tb)
	gen, err := LoadGeneratorCPU(modelPath, releasedConfig(tb))
	if err != nil {
		tb.Fatal(err)
	}
	voice, err := LoadVoiceState(voicePath, gen.FlowLM.Transformer, 64)
	if err != nil {
		tb.Fatal(err)
	}
	session, err := NewSession(gen, voice, frames)
	if err != nil {
		tb.Fatal(err)
	}
	noise := func(_ int, dst []float32) error {
		for i := range dst {
			dst[i] = float32((i*7)%19-9) / 16
		}
		return nil
	}
	return session, noise, make([]float32, frames*SamplesPerFrame)
}

func TestReleasedWarmGenerateZeroAlloc(t *testing.T) {
	session, noise, pcm := releasedWarmSession(t, 5)
	tokens := []uint32{2994, 578, 682}
	if _, err := session.GenerateInto(pcm, tokens, 5, 3, 1, -4, noise); err != nil {
		t.Fatal(err)
	}
	if allocs := testing.AllocsPerRun(10, func() {
		if _, err := session.GenerateInto(pcm, tokens, 5, 3, 1, -4, noise); err != nil {
			panic(err)
		}
	}); allocs != 0 {
		t.Fatalf("warm allocations=%g", allocs)
	}
}

func benchmarkReleasedFrames(b *testing.B, frames int) {
	session, noise, pcm := releasedWarmSession(b, frames)
	tokens := []uint32{2994, 578, 682}
	b.ReportAllocs()
	b.SetBytes(int64(frames * SamplesPerFrame * 4))
	b.ResetTimer()
	for b.Loop() {
		if _, err := session.GenerateInto(pcm, tokens, frames, 3, 1, -4, noise); err != nil {
			b.Fatal(err)
		}
	}
}
func BenchmarkReleasedFirstChunk(b *testing.B)       { benchmarkReleasedFrames(b, 1) }
func BenchmarkReleasedFiveFrames(b *testing.B)       { benchmarkReleasedFrames(b, 5) }
func BenchmarkReleasedTwentyFiveFrames(b *testing.B) { benchmarkReleasedFrames(b, 25) }
