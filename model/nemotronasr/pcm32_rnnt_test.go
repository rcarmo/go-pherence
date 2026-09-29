package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedPCM32RNNTJFKPrefixPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadPCM32RNNT(file)
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
	prefix := append([]float32(nil), pcm[:5160]...)
	initial := append([]float32(nil), prefix...)
	for _, lookahead := range []int{0, 3} {
		ref := readStemFixture(t, fmt.Sprintf("asr_composed_greedy_logits_look%d", lookahead), 5*rnntVocabulary)
		var maxAbs, sumAbs float64
		var outside, steps int
		tokens, frames, err := model.decode(prefix, lookahead, 101, func(step int, logits []float32) error {
			if step >= 5 {
				return fmt.Errorf("unexpected extra RNNT step")
			}
			steps++
			for i, value := range logits {
				want := ref[step*rnntVocabulary+i]
				delta := math.Abs(float64(value - want))
				maxAbs = math.Max(maxAbs, delta)
				sumAbs += delta
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(float64(want)) {
					outside++
				}
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		mean := sumAbs / float64(len(ref))
		t.Logf("lookahead=%d PCM-to-RNNT logits max_abs=%g mean_abs=%g outside=%d", lookahead, maxAbs, mean, outside)
		if steps != 5 || outside != 0 || mean > 5e-4 || !reflect.DeepEqual(tokens, []int{rnntBlank, rnntBlank, rnntBlank, rnntBlank, rnntBlank}) || !reflect.DeepEqual(frames, []int{0, 1, 2, 3, 4}) {
			t.Fatalf("PCM-to-RNNT mismatch tokens=%v frames=%v", tokens, frames)
		}
	}
	if !reflect.DeepEqual(prefix, initial) {
		t.Fatal("mutated PCM")
	}
}

func TestPCM32RNNTRejectsMalformed(t *testing.T) {
	if _, _, err := (*PCM32RNNT)(nil).Decode(make([]float32, 5160), 3, 101); err == nil {
		t.Fatal("accepted nil PCM decoder")
	}
	if _, _, err := (&PCM32RNNT{}).Decode(make([]float32, 5160), 3, 101); err == nil {
		t.Fatal("accepted missing weights")
	}
}
