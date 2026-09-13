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

type hubertFixture struct {
	Input    []float32 `json:"input"`
	Frames   int       `json:"frames"`
	Semantic []float32 `json:"semantic"`
}

const realHubertBenchmarkSamples = 32000

func realHubertModelPath() string { return os.Getenv("GO_PHERENCE_REAL_CODEC") }

func buildRealHubertInput(samples int) []float32 {
	wave := make([]float32, samples+320)
	for i := 0; i < samples; i++ {
		t := float64(i) / 16000.0
		wave[160+i] = float32(0.35*math.Sin(2*math.Pi*220.0*t) + 0.10*math.Cos(2*math.Pi*440.0*t) + 0.05*math.Sin(2*math.Pi*660.0*t+0.3))
	}
	return wave
}

func loadRealHubert2s(tb testing.TB) (*Hubert, []float32, int) {
	tb.Helper()
	modelPath := realHubertModelPath()
	if _, err := os.Stat(filepath.Join(modelPath, "model.safetensors")); err != nil {
		tb.Skip("set GO_PHERENCE_REAL_CODEC to a HiggsAudioV2 tokenizer directory")
	}
	weights, err := loader.LoadHubert(modelPath)
	if err != nil {
		tb.Fatal(err)
	}
	hubert, err := NewHubert(weights)
	if err != nil {
		tb.Fatal(err)
	}
	input := buildRealHubertInput(realHubertBenchmarkSamples)
	frames, err := hubert.Prepare(len(input))
	if err != nil {
		tb.Fatal(err)
	}
	return hubert, input, frames
}

func TestRealHubertParity(t *testing.T) {
	modelPath := os.Getenv("GO_PHERENCE_REAL_CODEC")
	if modelPath == "" {
		t.Skip("set GO_PHERENCE_REAL_CODEC")
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
	script := filepath.Clean(filepath.Join(wd, "../../scripts/omnivoice-hubert-fixture.py"))
	if _, err = os.Stat(script); err != nil {
		t.Fatalf("missing fixture script: %v", err)
	}
	fixturePath := filepath.Join(t.TempDir(), "hubert-fixture.json")
	cmd := exec.Command(python, script, "--model", modelPath, "--out", fixturePath)
	cmd.Env = os.Environ()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("fixture generator: %v\n%s", err, out)
	}
	raw, err := os.ReadFile(fixturePath)
	if err != nil {
		t.Fatal(err)
	}
	var fixture hubertFixture
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	weights, err := loader.LoadHubert(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	hubert, err := NewHubert(weights)
	if err != nil {
		t.Fatal(err)
	}
	frames, err := hubert.Prepare(len(fixture.Input))
	if err != nil {
		t.Fatal(err)
	}
	if frames != fixture.Frames {
		t.Fatalf("frames=%d want %d", frames, fixture.Frames)
	}
	got := make([]float32, frames*hubertHidden)
	ctx := context.Background()
	if err := hubert.ExtractInto(ctx, got, fixture.Input); err != nil {
		t.Fatal(err)
	}
	if len(got) != len(fixture.Semantic) {
		t.Fatalf("semantic=%d want %d", len(got), len(fixture.Semantic))
	}
	maxErr, rms := 0.0, 0.0
	for i, want := range fixture.Semantic {
		err := math.Abs(float64(got[i] - want))
		if err > maxErr {
			maxErr = err
		}
		rms += err * err
	}
	if len(got) > 0 {
		rms = math.Sqrt(rms / float64(len(got)))
	}
	if maxErr > 1e-3 {
		t.Fatalf("max semantic error=%g rms=%g", maxErr, rms)
	}
	var allocErr error
	allocs := testing.AllocsPerRun(5, func() {
		allocErr = hubert.ExtractInto(ctx, got, fixture.Input)
	})
	if allocErr != nil {
		t.Fatal(allocErr)
	}
	if allocs != 0 {
		t.Fatalf("ExtractInto allocations=%g want 0", allocs)
	}
	t.Logf("hubert parity ok frames=%d max_err=%g rms=%g; ExtractInto allocations=%g", frames, maxErr, rms, allocs)
}

func TestRealHubertPackedParity2s(t *testing.T) {
	hubert, input, frames := loadRealHubert2s(t)
	before := make([]float32, frames*hubertHidden)
	after := make([]float32, len(before))
	ctx := context.Background()
	if err := hubert.extractInto(ctx, before, input, false); err != nil {
		t.Fatal(err)
	}
	if err := hubert.extractInto(ctx, after, input, true); err != nil {
		t.Fatal(err)
	}
	maxErr, rms := 0.0, 0.0
	for i, want := range before {
		err := math.Abs(float64(after[i] - want))
		if err > maxErr {
			maxErr = err
		}
		rms += err * err
	}
	if len(after) > 0 {
		rms = math.Sqrt(rms / float64(len(after)))
	}
	if maxErr > 1e-3 {
		t.Fatalf("packed parity max error=%g rms=%g", maxErr, rms)
	}
	var allocErr error
	allocs := testing.AllocsPerRun(5, func() {
		allocErr = hubert.ExtractInto(ctx, after, input)
	})
	if allocErr != nil {
		t.Fatal(allocErr)
	}
	if allocs != 0 {
		t.Fatalf("ExtractInto allocations=%g want 0", allocs)
	}
	t.Logf("hubert packed parity ok samples=%d frames=%d max_err=%g rms=%g; ExtractInto allocations=%g", len(input), frames, maxErr, rms, allocs)
}

func BenchmarkRealHubertExtract2s(b *testing.B) {
	hubert, input, frames := loadRealHubert2s(b)
	ctx := context.Background()
	b.Run("before_nt", func(b *testing.B) {
		dst := make([]float32, frames*hubertHidden)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := hubert.extractInto(ctx, dst, input, false); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("after_packed", func(b *testing.B) {
		dst := make([]float32, frames*hubertHidden)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := hubert.ExtractInto(ctx, dst, input); err != nil {
				b.Fatal(err)
			}
		}
	})
}
