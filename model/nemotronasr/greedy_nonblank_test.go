package nemotronasr

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// The five projected rows come from the independently released PyTorch
// full-recording JFK encoder. Only the native RNNT decoder is checked here.
func TestReleasedGreedyRNNTJFKNonblankPyTorchParity(t *testing.T) {
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
	fixture, err := os.ReadFile("testdata/asr_jfk_nonblank.json")
	if err != nil {
		t.Fatal(err)
	}
	var want struct {
		EncoderStart int   `json:"encoder_start"`
		EncoderRows  int   `json:"encoder_rows"`
		Lookahead    int   `json:"lookahead"`
		PromptID     int   `json:"prompt_id"`
		Tokens       []int `json:"tokens"`
		Frames       []int `json:"frames"`
	}
	if err := json.Unmarshal(fixture, &want); err != nil {
		t.Fatal(err)
	}
	if want.EncoderStart != 10 || want.EncoderRows != 5 || want.Lookahead != 3 || want.PromptID != 101 || len(want.Tokens) != len(want.Frames) || len(want.Tokens) == 0 || len(want.Tokens) > want.EncoderRows*10 {
		t.Fatal("invalid JFK nonblank fixture metadata")
	}
	encoder := readStemFixture(t, "asr_jfk_nonblank_encoder", want.EncoderRows*rnntHidden)
	initial := append([]float32(nil), encoder...)
	ref := readStemFixture(t, "asr_jfk_nonblank_logits", len(want.Tokens)*rnntVocabulary)
	var maxAbs, sumAbs float64
	var outside, steps int
	tokens, frames, err := (&GreedyRNNT{Decoder: decoder, Projection: projection}).decode(encoder, want.EncoderRows, func(step int, got []float32) error {
		if step >= len(want.Tokens) {
			return fmt.Errorf("too many RNNT steps")
		}
		steps++
		for i, value := range got {
			base := float64(ref[step*rnntVocabulary+i])
			delta := math.Abs(float64(value) - base)
			maxAbs = math.Max(maxAbs, delta)
			sumAbs += delta
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || delta > 3e-4+2e-5*math.Abs(base) {
				outside++
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	mean := sumAbs / float64(len(ref))
	t.Logf("nonblank JFK logits max_abs=%g mean_abs=%g outside=%d", maxAbs, mean, outside)
	if steps != len(want.Tokens) || outside != 0 || mean > 5e-4 {
		t.Fatal("RNNT logits differ from PyTorch")
	}
	if !reflect.DeepEqual(tokens, want.Tokens) || !reflect.DeepEqual(frames, want.Frames) {
		t.Fatalf("tokens=%v frames=%v, want tokens=%v frames=%v", tokens, frames, want.Tokens, want.Frames)
	}
	nonblank := 0
	for _, token := range tokens {
		if token != rnntBlank {
			nonblank++
		}
	}
	if nonblank < 2 || !reflect.DeepEqual(encoder, initial) {
		t.Fatal("missing nonblank emission or mutated encoder")
	}
}
