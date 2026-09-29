package nemotrondiarization

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedOfflineRequestPyTorchLogitParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadOfflineRequest(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) != 176000 {
		t.Fatalf("JFK input rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	original := append([]float32(nil), pcm...)
	got, frames, err := model.ForwardPCM(pcm)
	if err != nil {
		t.Fatal(err)
	}
	if frames != 1101 || len(got) != 1101*diarizationSpeakers {
		t.Fatalf("frames=%d logits=%d", frames, len(got))
	}
	ref := readStackingFixture(t, "testdata/jfk_request_logits.f32.gz", len(got))
	var maxAbs, sumAbs float64
	var outside int
	for i, value := range got {
		delta := math.Abs(float64(value - ref[i]))
		maxAbs = math.Max(maxAbs, delta)
		sumAbs += delta
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
			outside++
		}
	}
	mean := sumAbs / float64(len(got))
	t.Logf("full PCM request max_abs=%g mean_abs=%g outside=%d", maxAbs, mean, outside)
	if outside != 0 || mean > 1e-5 {
		t.Fatal("full PCM request differs from independent PyTorch")
	}
	for i, value := range pcm {
		if value != original[i] {
			t.Fatalf("mutated PCM %d", i)
		}
	}
}

func BenchmarkReleasedOfflineRequestJFK(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		b.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		b.Fatal(err)
	}
	model, err := LoadOfflineRequest(file)
	if err != nil {
		b.Fatal(err)
	}
	if err := file.Close(); err != nil {
		b.Fatal(err)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 {
		b.Fatalf("WAV rate=%d err=%v", rate, err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, _, err := model.ForwardPCM(pcm); err != nil {
			b.Fatal(err)
		}
	}
}

func TestOfflineRequestRejectsMalformed(t *testing.T) {
	if _, err := LoadOfflineRequest(nil); err == nil {
		t.Fatal("accepted nil checkpoint")
	}
	if _, _, err := (*OfflineRequest)(nil).ForwardPCM(make([]float32, 16000)); err == nil {
		t.Fatal("accepted nil request")
	}
	bad := make([]float32, 16000)
	bad[0] = float32(math.NaN())
	if _, _, err := (&OfflineRequest{stacking: &StackingProjection{}, tower: &OfflineAudioTower{}, head: &OfflineHead{}}).ForwardPCM(bad); err == nil {
		t.Fatal("accepted non-finite PCM")
	}
}
