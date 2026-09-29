package nemotronasr

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedOfflinePCM32PyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadOfflineProjection(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 || len(pcm) != 176000 {
		t.Fatalf("JFK rate=%d samples=%d err=%v", rate, len(pcm), err)
	}
	prefix := append([]float32(nil), pcm[:31*160+200]...)
	original := append([]float32(nil), prefix...)
	for _, lookahead := range []int{0, 3} {
		tower, encoder, err := model.ForwardPCM32(prefix, lookahead, 101)
		if err != nil {
			t.Fatal(err)
		}
		for _, item := range []struct {
			name string
			got  []float32
		}{{"tower", tower}, {"encoder", encoder}} {
			// The PyTorch processor crop of the full JFK fixture is independently
			// checked by scripts/nemotron_asr_composed_fixture.py.
			ref := readStemFixture(t, "asr_composed_"+item.name+"_look"+string(rune('0'+lookahead)), len(item.got))
			var maxAbs, sumAbs float64
			var outside int
			for i, value := range item.got {
				delta := math.Abs(float64(value - ref[i]))
				maxAbs = math.Max(maxAbs, delta)
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(ref[i])) {
					outside++
				}
			}
			t.Logf("PCM lookahead=%d %s max_abs=%g mean_abs=%g outside=%d", lookahead, item.name, maxAbs, sumAbs/float64(len(item.got)), outside)
			if outside != 0 {
				t.Fatal("PCM projection differs from PyTorch")
			}
		}
	}
	for i, value := range prefix {
		if value != original[i] {
			t.Fatalf("mutated PCM %d", i)
		}
	}
}

func TestOfflinePCM32RejectsMalformed(t *testing.T) {
	if _, _, err := (*OfflineProjection)(nil).ForwardPCM32(make([]float32, 31*160+200), 0, 101); err == nil {
		t.Fatal("accepted nil model")
	}
	if _, _, err := (&OfflineProjection{}).ForwardPCM32(make([]float32, 31*160+199), 0, 101); err == nil {
		t.Fatal("accepted short PCM")
	}
}
