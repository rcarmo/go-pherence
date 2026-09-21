package jevlike

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
	backbone "github.com/rcarmo/go-pherence/model"
)

func TestPublishedFrozenQwenParity(t *testing.T) {
	dir := os.Getenv("JEVLIKE_FROZEN_MODEL_DIR")
	if dir == "" {
		t.Skip("set JEVLIKE_FROZEN_MODEL_DIR to Qwen2.5-0.5B checkpoint")
	}
	var ref struct {
		Examples   []ChoiceExample
		Checkpoint Checkpoint
		ContextIDs [][]int `json:"context_ids"`
		Hidden     [][][]float32
		Logits     [][]float32
	}
	raw, err := os.ReadFile("testdata/frozen_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := backbone.LoadLlama(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	encoder := DecoderEncoder{Model: m, Tokenize: func(text string) ([]int, error) { return tok.Encode(text), nil }}
	scorer, err := ref.Checkpoint.Frozen(encoder)
	if err != nil {
		t.Fatal(err)
	}
	var maxHidden float64
	for i, ex := range ref.Examples {
		ids := tok.Encode(ex.Context)
		if !reflect.DeepEqual(ids, ref.ContextIDs[i]) {
			t.Fatalf("token IDs got%v want%v", ids, ref.ContextIDs[i])
		}
		hidden, err := encoder.Encode(ex.Context, 64)
		if err != nil {
			t.Fatal(err)
		}
		if len(hidden) != len(ref.Hidden[i]) {
			t.Fatal("hidden row count")
		}
		for row, values := range hidden {
			for col, v := range values {
				d := math.Abs(float64(v - ref.Hidden[i][row][col]))
				maxHidden = math.Max(maxHidden, d)
				if d > 2e-3 {
					t.Fatalf("hidden %d,%d,%d got%g want%g delta%g", i, row, col, v, ref.Hidden[i][row][col], d)
				}
			}
		}
	}
	logits, err := scorer.Forward(ref.Examples, false)
	if err != nil {
		t.Fatal(err)
	}
	var maxLogit float64
	for row, ex := range ref.Examples {
		for col := range ex.Options {
			d := math.Abs(float64(logits[row][col] - ref.Logits[row][col]))
			maxLogit = math.Max(maxLogit, d)
			if d > 2e-4 {
				t.Fatalf("logit %d,%d got%g want%g", row, col, logits[row][col], ref.Logits[row][col])
			}
		}
	}
	t.Logf("Qwen frozen hidden max delta=%g scorer logits max delta=%g", maxHidden, maxLogit)
}
