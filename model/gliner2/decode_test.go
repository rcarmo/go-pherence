package gliner2

import (
	"math"
	"strings"
	"testing"
	"unicode/utf8"
)

type decodeRow struct {
	Start int
	End   int
	Probs []float64
}

func TestDecodeEntitiesFlatWeightedIntervalBeatsGreedy(t *testing.T) {
	text := "aa bb cc dd"
	scores := makeDecodeScores(t, text, []string{"item"}, []decodeRow{
		{Start: 0, End: 4, Probs: []float64{0.9}},
		{Start: 0, End: 2, Probs: []float64{0.6}},
		{Start: 2, End: 4, Probs: []float64{0.6}},
	})

	got, err := DecodeEntities(text, scores, 0.5, "flat")
	if err != nil {
		t.Fatal(err)
	}
	assertEntitySpans(t, got, [][2]int{{0, 2}, {2, 4}})
}

func TestDecodeEntitiesFlatTieBreakIsDeterministic(t *testing.T) {
	text := "aa bb cc dd"
	scores := makeDecodeScores(t, text, []string{"item"}, []decodeRow{
		{Start: 0, End: 2, Probs: []float64{0.7}},
		{Start: 2, End: 4, Probs: []float64{0.3}},
		{Start: 0, End: 1, Probs: []float64{0.6}},
		{Start: 1, End: 4, Probs: []float64{0.4}},
	})

	got, err := DecodeEntities(text, scores, 0, "disallow")
	if err != nil {
		t.Fatal(err)
	}
	assertEntitySpans(t, got, [][2]int{{0, 2}, {2, 4}})
}

func TestDecodeEntitiesNestedAliasCollapsesDuplicates(t *testing.T) {
	text := "zero one two three"
	scores := makeDecodeScores(t, text, []string{"item"}, []decodeRow{
		{Start: 0, End: 3, Probs: []float64{0.7}},
		{Start: 0, End: 3, Probs: []float64{0.9}},
		{Start: 1, End: 2, Probs: []float64{0.8}},
		{Start: 2, End: 4, Probs: []float64{0.85}},
	})

	got, err := DecodeEntities(text, scores, 0.5, "allow_nested")
	if err != nil {
		t.Fatal(err)
	}
	assertEntitySpans(t, got, [][2]int{{0, 3}, {1, 2}})
	if math.Abs(got[0].Confidence-0.9) > 1e-6 {
		t.Fatalf("duplicate collapse kept confidence %g want 0.9", got[0].Confidence)
	}
}

func TestDecodeEntitiesRanksEqualTiesByLabel(t *testing.T) {
	text := "Ada"
	scores := makeDecodeScores(t, text, []string{"zeta", "alpha"}, []decodeRow{{Start: 0, End: 1, Probs: []float64{0.8, 0.8}}})

	got, err := DecodeEntities(text, scores, 0.5, "allow")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("len=%d want=2", len(got))
	}
	if got[0].Label != "alpha" || got[1].Label != "zeta" {
		t.Fatalf("labels=%v want [alpha zeta]", []string{got[0].Label, got[1].Label})
	}
}

func TestDecodeEntitiesUnicodeOffsetsAndThresholdEquality(t *testing.T) {
	text := "Olá 世界"
	scores := makeDecodeScores(t, text, []string{"thing"}, []decodeRow{
		{Start: 0, End: 2, Probs: []float64{0.75}},
		{Start: 0, End: 1, Probs: []float64{0.74}},
	})

	got, err := DecodeEntities(text, scores, 0.75, "allow")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("len=%d want=1", len(got))
	}
	entity := got[0]
	if entity.Text != text {
		t.Fatalf("text=%q want %q", entity.Text, text)
	}
	if entity.ByteStart != 0 || entity.ByteEnd != len(text) {
		t.Fatalf("byte span=[%d,%d) want [0,%d)", entity.ByteStart, entity.ByteEnd, len(text))
	}
	if entity.CharStart != 0 || entity.CharEnd != utf8.RuneCountInString(text) {
		t.Fatalf("char span=[%d,%d) want [0,%d)", entity.CharStart, entity.CharEnd, utf8.RuneCountInString(text))
	}
	if entity.TokenStart != 0 || entity.TokenEnd != 2 {
		t.Fatalf("token span=[%d,%d) want [0,2)", entity.TokenStart, entity.TokenEnd)
	}
	if math.Abs(entity.Confidence-0.75) > 1e-6 {
		t.Fatalf("confidence=%g want 0.75", entity.Confidence)
	}
	if text[entity.ByteStart:entity.ByteEnd] != entity.Text {
		t.Fatalf("byte slice=%q want %q", text[entity.ByteStart:entity.ByteEnd], entity.Text)
	}
}

func TestDecodeEntitiesRejectsInvalidWordOffsets(t *testing.T) {
	text := "Olá mundo"
	words, err := SplitWords(text)
	if err != nil {
		t.Fatal(err)
	}
	words[1].Start++
	scores := EntityScores{
		Input: EntityInput{Words: words, Labels: []string{"thing"}},
		Candidates: PooledCandidates{
			Indices:   [][]int{{0, 2}},
			ValidMask: []bool{true},
		},
		Logits: [][]float32{{0}},
	}

	_, err = DecodeEntities(text, scores, 0.5, "allow")
	if err == nil || !strings.Contains(err.Error(), "char start") {
		t.Fatalf("err=%v want char start validation", err)
	}
}

func TestDecodeEntitiesRejectsNonFiniteLogit(t *testing.T) {
	text := "Ada Lovelace"
	words, err := SplitWords(text)
	if err != nil {
		t.Fatal(err)
	}
	scores := EntityScores{
		Input: EntityInput{Words: words, Labels: []string{"person"}},
		Candidates: PooledCandidates{
			Indices:   [][]int{{0, 2}},
			ValidMask: []bool{true},
		},
		Logits: [][]float32{{float32(math.Inf(1))}},
	}

	_, err = DecodeEntities(text, scores, 0.5, "allow")
	if err == nil || !strings.Contains(err.Error(), "non-finite logit") {
		t.Fatalf("err=%v want non-finite logit validation", err)
	}
}

func makeDecodeScores(t *testing.T, text string, labels []string, rows []decodeRow) EntityScores {
	t.Helper()
	words, err := SplitWords(text)
	if err != nil {
		t.Fatal(err)
	}
	indices := make([][]int, len(rows))
	valid := make([]bool, len(rows))
	logits := make([][]float32, len(rows))
	for i, row := range rows {
		if len(row.Probs) != len(labels) {
			t.Fatalf("row %d has %d probs want %d", i, len(row.Probs), len(labels))
		}
		indices[i] = []int{row.Start, row.End}
		valid[i] = true
		logits[i] = make([]float32, len(labels))
		for j, probability := range row.Probs {
			logits[i][j] = float32(logit(probability))
		}
	}
	return EntityScores{
		Input: EntityInput{Words: words, Labels: append([]string(nil), labels...)},
		Candidates: PooledCandidates{
			Indices:   indices,
			ValidMask: valid,
		},
		Logits: logits,
	}
}

func logit(p float64) float64 {
	return math.Log(p / (1 - p))
}

func assertEntitySpans(t *testing.T, entities []Entity, want [][2]int) {
	t.Helper()
	got := make([][2]int, len(entities))
	for i, entity := range entities {
		got[i] = [2]int{entity.TokenStart, entity.TokenEnd}
	}
	if len(got) != len(want) {
		t.Fatalf("spans=%v want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("spans=%v want %v", got, want)
		}
	}
}
