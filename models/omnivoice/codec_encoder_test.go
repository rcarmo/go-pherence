package omnivoice

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

type codecEncoderFixture struct {
	Waveform       []float32 `json:"waveform"`
	SemanticFrames int       `json:"semantic_frames"`
	Semantic       []float32 `json:"semantic"`
	Codes          []int     `json:"codes"`
	Books          int       `json:"books"`
	Frames         int       `json:"frames"`
}

func TestRealCodecEncoderParity(t *testing.T) {
	modelPath := os.Getenv("GO_PHERENCE_REAL_CODEC")
	if modelPath == "" {
		modelPath = "/workspace/projects/spock-tts/models/omnivoice/audio_tokenizer"
	}
	if _, err := os.Stat(filepath.Join(modelPath, "model.safetensors")); err != nil {
		t.Skip("set GO_PHERENCE_REAL_CODEC to a HiggsAudioV2 tokenizer directory")
	}
	python := os.Getenv("GO_PHERENCE_REFERENCE_PYTHON")
	if python == "" {
		t.Skip("set GO_PHERENCE_REFERENCE_PYTHON")
	}
	if _, err := os.Stat(python); err != nil {
		t.Skip("python fixture generator unavailable")
	}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	script := filepath.Clean(filepath.Join(wd, "../../scripts/omnivoice-encoder-fixture.py"))
	if _, err = os.Stat(script); err != nil {
		t.Fatalf("missing fixture script: %v", err)
	}
	fixturePath := filepath.Join(t.TempDir(), "codec-encoder-fixture.json")
	cmd := exec.Command(python, script, "--model", modelPath, "--out", fixturePath)
	cmd.Env = os.Environ()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fixture generator: %v\n%s", err, out)
	}
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fixture codecEncoderFixture
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Books != 8 || fixture.Frames != fixture.SemanticFrames {
		t.Fatalf("bad fixture dimensions books=%d frames=%d semantic=%d", fixture.Books, fixture.Frames, fixture.SemanticFrames)
	}
	weights, err := loader.LoadCodecEncoder(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	encoder, err := NewCodecEncoder(weights)
	if err != nil {
		t.Fatal(err)
	}
	if err := encoder.Prepare(len(fixture.Waveform), fixture.SemanticFrames); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	codes := make([]int, len(fixture.Codes))
	if err := encoder.EncodeFeaturesInto(ctx, codes, fixture.Waveform, fixture.Semantic, fixture.SemanticFrames); err != nil {
		t.Fatal(err)
	}
	if len(codes) != len(fixture.Codes) {
		t.Fatalf("codes=%d want %d", len(codes), len(fixture.Codes))
	}
	for i, got := range codes {
		if got != fixture.Codes[i] {
			t.Fatalf("code %d got %d want %d", i, got, fixture.Codes[i])
		}
	}
	var allocErr error
	allocs := testing.AllocsPerRun(5, func() {
		allocErr = encoder.EncodeFeaturesInto(ctx, codes, fixture.Waveform, fixture.Semantic, fixture.SemanticFrames)
	})
	if allocErr != nil {
		t.Fatal(allocErr)
	}
	if allocs != 0 {
		t.Fatalf("EncodeFeaturesInto allocations=%g want 0", allocs)
	}
	t.Logf("encoder parity ok for %d books x %d frames; EncodeFeaturesInto allocations=%g", fixture.Books, fixture.Frames, allocs)
}

func codecEncoderModelPath(tb testing.TB) string {
	tb.Helper()
	modelPath := os.Getenv("GO_PHERENCE_REAL_CODEC")
	if modelPath == "" {
		modelPath = "/workspace/projects/spock-tts/models/omnivoice/audio_tokenizer"
	}
	if _, err := os.Stat(filepath.Join(modelPath, "model.safetensors")); err != nil {
		tb.Skip("set GO_PHERENCE_REAL_CODEC to a HiggsAudioV2 tokenizer directory")
	}
	return modelPath
}

func newBenchmarkCodecEncoder(tb testing.TB) *CodecEncoder {
	tb.Helper()
	weights, err := loader.LoadCodecEncoder(codecEncoderModelPath(tb))
	if err != nil {
		tb.Fatal(err)
	}
	encoder, err := NewCodecEncoder(weights)
	if err != nil {
		tb.Fatal(err)
	}
	return encoder
}

func codecEncoderBuildWave(samples int) []float32 {
	wave := make([]float32, samples)
	for i := range wave {
		t := float64(i) / 24000.0
		v := 0.35 * math.Sin(2*math.Pi*220.0*t)
		v += 0.10 * math.Cos(2*math.Pi*440.0*t)
		v += 0.05 * math.Sin(2*math.Pi*660.0*t+0.3)
		wave[i] = float32(v)
	}
	return wave
}

func codecEncoderBuildSemantic(frames int) []float32 {
	semantic := make([]float32, frames*768)
	for i := range semantic {
		x := float64(i)
		v := 0.50 * math.Sin(x*0.017)
		v += 0.25 * math.Cos(x*0.011+0.2)
		v += 0.10 * math.Sin(x*0.007+0.5)
		semantic[i] = float32(v)
	}
	return semantic
}

func codecEncoderSamplesForFrames(tb testing.TB, encoder *CodecEncoder, frames int) int {
	tb.Helper()
	for samples := frames * 600; samples <= frames*1200; samples++ {
		if encoder.Prepare(samples, frames) == nil {
			return samples
		}
	}
	tb.Fatalf("no sample count found for %d frames", frames)
	return 0
}

func codecEncoderFramesForSamples(tb testing.TB, encoder *CodecEncoder, samples int) int {
	tb.Helper()
	for frames := 1; frames <= 500; frames++ {
		if encoder.Prepare(samples, frames) == nil {
			return frames
		}
	}
	tb.Fatalf("no semantic frame count found for %d samples", samples)
	return 0
}

func BenchmarkCodecEncoderEncodeFeaturesIntoSynthetic50Frames(b *testing.B) {
	encoder := newBenchmarkCodecEncoder(b)
	frames := 50
	samples := codecEncoderSamplesForFrames(b, encoder, frames)
	wave := codecEncoderBuildWave(samples)
	semantic := codecEncoderBuildSemantic(frames)
	codes := make([]int, encoder.weights.Quantizers*frames)
	ctx := context.Background()
	if err := encoder.Prepare(samples, frames); err != nil {
		b.Fatal(err)
	}
	if err := encoder.EncodeFeaturesInto(ctx, codes, wave, semantic, frames); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := encoder.EncodeFeaturesInto(ctx, codes, wave, semantic, frames); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCodecEncoderEncodeFeaturesIntoSynthetic2Seconds(b *testing.B) {
	encoder := newBenchmarkCodecEncoder(b)
	samples := 2 * encoder.weights.SampleRate
	frames := codecEncoderFramesForSamples(b, encoder, samples)
	wave := codecEncoderBuildWave(samples)
	semantic := codecEncoderBuildSemantic(frames)
	codes := make([]int, encoder.weights.Quantizers*frames)
	ctx := context.Background()
	if err := encoder.Prepare(samples, frames); err != nil {
		b.Fatal(err)
	}
	if err := encoder.EncodeFeaturesInto(ctx, codes, wave, semantic, frames); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := encoder.EncodeFeaturesInto(ctx, codes, wave, semantic, frames); err != nil {
			b.Fatal(err)
		}
	}
}
