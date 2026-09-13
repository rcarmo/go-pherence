package whisper

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestAlignmentCostAndDTWPinnedOracle(t *testing.T) {
	var f struct {
		Heads, Text, Frames      int
		Input                    []float32
		Matrix                   []float64
		TextIndices, TimeIndices []int `json:"-"`
		RawTextIndices           []int `json:"text_indices"`
		RawTimeIndices           []int `json:"time_indices"`
	}
	data, err := os.ReadFile("testdata/alignment-matrix-oracle.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	got, err := alignmentCost(context.Background(), append([]float32(nil), f.Input...), f.Heads, f.Text, f.Frames, 7)
	if err != nil || len(got) != len(f.Matrix) {
		t.Fatal(err, len(got))
	}
	for i, value := range got {
		if math.Abs(value-f.Matrix[i]) > 2e-6 {
			t.Fatalf("matrix[%d]=%.9g want %.9g", i, value, f.Matrix[i])
		}
	}
	text, times, err := dynamicTimeWarp(context.Background(), got, f.Text, f.Frames)
	if err != nil || !reflect.DeepEqual(text, f.RawTextIndices) || !reflect.DeepEqual(times, f.RawTimeIndices) {
		t.Fatal("DTW mismatch", err, text, times)
	}
}

func TestWordGroupsPreserveWhitespaceAndUTF8(t *testing.T) {
	tok := &Tokenizer{Vocab: map[int]string{1: "Hello", 2: "Ġworld", 3: "!", 4: "ĠTelefÃ", 5: "³nica"}, VocabSize: 6}
	got, err := tok.wordGroups([]int{1, 2, 3, 4, 5})
	if err != nil {
		t.Fatal(err)
	}
	want := []wordGroup{{"Hello", 0, 1}, {"world!", 1, 3}, {"Telefónica", 3, 5}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("groups=%+v want %+v", got, want)
	}
}

func TestAlignWordsCheckedValidation(t *testing.T) {
	ctx := context.Background()
	if _, err := AlignWordsChecked(ctx, nil, nil, nil, nil, "en", []int{1}, 2); !errors.Is(err, ErrWordAlignmentInput) {
		t.Fatal(err)
	}
	if _, _, err := dynamicTimeWarp(ctx, []float64{math.NaN()}, 1, 1); !errors.Is(err, ErrWordAlignmentInput) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, _, err := dynamicTimeWarp(ctx, []float64{0}, 1, 1); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
}
