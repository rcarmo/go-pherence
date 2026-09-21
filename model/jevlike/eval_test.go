package jevlike

import (
	"math"
	"testing"
)

func TestMetricsKnown(t *testing.T) {
	m, err := ComputeMetrics([][]float32{{0, 0}, {1000, 0}, {0, 1, 2, 3}}, []int{0, 1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if m.Top1 != 1.0/3 || m.Top3 != 2.0/3 || m.Examples != 3 {
		t.Fatalf("metrics %+v", m)
	}
	confidence := float64(softmaxFloat32([]float32{0, 1, 2, 3})[3])
	want := (0.5 + 1 + confidence) / 3
	if math.Abs(m.ECE-want) > 1e-7 {
		t.Fatalf("ece %g want %g", m.ECE, want)
	}
}
func TestShuffledContextRoll(t *testing.T) {
	m, err := NewInitializedTinyScorer(Config{Width: 4, Rank: 3, ContextTokens: 8, OptionTokens: 4}, 7)
	if err != nil {
		t.Fatal(err)
	}
	examples := []ChoiceExample{{Context: "abc", Options: []string{"a", "b"}, Label: 0}, {Context: "d", Options: []string{"cd", "a"}, Label: 1}}
	batch, _ := BuildByteBatch(examples, 8, 4)
	got, err := m.ForwardShuffled(batch)
	if err != nil {
		t.Fatal(err)
	}
	swapped := append([]ChoiceExample(nil), examples...)
	swapped[0].Context, swapped[1].Context = examples[1].Context, examples[0].Context
	other, _ := BuildByteBatch(swapped, 8, 4)
	want, err := m.Forward(other)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		for j := range got[i] {
			if got[i][j] != want[i][j] {
				t.Fatal("context roll mismatch")
			}
		}
	}
	if _, err := EvaluateTiny(m, examples, 2); err != nil {
		t.Fatal(err)
	}
	one, _ := BuildByteBatch(examples[:1], 8, 4)
	a, _ := m.Forward(one)
	b, _ := m.ForwardShuffled(one)
	for i := range a[0] {
		if a[0][i] != b[0][i] {
			t.Fatal("singleton shuffled")
		}
	}
}
