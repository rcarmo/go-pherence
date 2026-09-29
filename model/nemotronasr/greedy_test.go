package nemotronasr

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedGreedyRNNTFiveFramePyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadRNNTDecoder(file)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := LoadRNNTProjection(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	greedy := &GreedyRNNT{Decoder: decoder, Projection: projection}
	for _, lookahead := range []int{0, 3} {
		encoder := readStemFixture(t, fmt.Sprintf("asr_composed_encoder_look%d", lookahead), 5*rnntHidden)
		original := append([]float32(nil), encoder...)
		ref := readStemFixture(t, fmt.Sprintf("asr_composed_greedy_logits_look%d", lookahead), 5*rnntVocabulary)
		var maxAbs, sumAbs float64
		var outside int
		tokens, frames, err := greedy.decode(encoder, 5, func(step int, got []float32) error {
			if step >= 5 {
				return fmt.Errorf("too many greedy steps")
			}
			for i, value := range got {
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
		t.Logf("lookahead=%d logits max_abs=%g mean_abs=%g outside=%d", lookahead, maxAbs, sumAbs/float64(len(ref)), outside)
		mean := sumAbs / float64(len(ref))
		if outside != 0 || mean > 5e-4 {
			t.Fatal("greedy logits differ from PyTorch")
		}
		if !reflect.DeepEqual(tokens, []int{rnntBlank, rnntBlank, rnntBlank, rnntBlank, rnntBlank}) || !reflect.DeepEqual(frames, []int{0, 1, 2, 3, 4}) {
			t.Fatalf("tokens=%v frames=%v", tokens, frames)
		}
		for i, value := range encoder {
			if value != original[i] {
				t.Fatalf("mutated encoder input %d", i)
			}
		}
	}
}

func TestGreedyRNNTTransitions(t *testing.T) {
	frame, symbols := 0, 0
	advance := func(token int) {
		if advanceGreedyFrame(token, &symbols) {
			frame++
		}
	}
	advance(3)
	if frame != 0 || symbols != 1 {
		t.Fatal("nonblank should remain on the frame")
	}
	for i := 0; i < 8; i++ {
		advance(3)
	}
	if frame != 0 || symbols != 9 {
		t.Fatal("advanced before symbol limit")
	}
	advance(3)
	if frame != 1 || symbols != 0 {
		t.Fatal("failed forced advance at ten symbols")
	}
	advance(rnntBlank)
	if frame != 2 || symbols != 0 {
		t.Fatal("blank failed to advance")
	}
}

func TestGreedyRNNTRejectsMalformed(t *testing.T) {
	if _, _, err := (*GreedyRNNT)(nil).Decode(make([]float32, rnntHidden), 1); err == nil {
		t.Fatal("accepted nil greedy model")
	}
	if _, _, err := (&GreedyRNNT{}).Decode(make([]float32, rnntHidden), 1); err == nil {
		t.Fatal("accepted missing weights")
	}
}
