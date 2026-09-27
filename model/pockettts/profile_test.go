package pockettts

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"testing"
)

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

// Opt-in full 25-frame PCM regression for the pinned released preset voice.
// The existing amd64 and ARM64 scalar paths have distinct baseline hashes;
// this checks that a kernel change preserves the result on its own host.
func TestReleasedTwentyFiveFrameFingerprint(t *testing.T) {
	if os.Getenv("GO_PHERENCE_POCKETTTS_FINGERPRINT") == "" {
		t.Skip("set GO_PHERENCE_POCKETTTS_FINGERPRINT for pinned released output")
	}
	session, noise, pcm := releasedWarmSession(t, 25)
	if _, err := session.GenerateInto(pcm, []uint32{2994, 578, 682}, 25, 3, 1, -4, noise); err != nil {
		t.Fatal(err)
	}
	h := sha256.New()
	var word [4]byte
	for _, v := range pcm {
		binary.LittleEndian.PutUint32(word[:], math.Float32bits(v))
		_, _ = h.Write(word[:])
	}
	t.Logf("released 25-frame PCM F32 SHA-256 %x; samples=%d", h.Sum(nil), len(pcm))
	pinned := map[string]string{
		"amd64": "e88c50c2938246054796c6369e263de7b2243eb2b6f87e6af209399f89bcc882",
		"arm64": "3309ea59f22f0b68c2c7646b099b968a5d85db03074d1c321440e6588fe11649",
	}[runtime.GOARCH]
	if pinned == "" {
		t.Skip("no native released fingerprint for " + runtime.GOARCH)
	}
	if fmt.Sprintf("%x", h.Sum(nil)) != pinned {
		t.Fatalf("released waveform fingerprint mismatch")
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
