package jevlike

import (
	"math"
	"testing"
)

func TestAttentionHeadReferenceAndPermutation(t *testing.T) {
	h, err := NewAttentionHead(2, 2)
	if err != nil {
		t.Fatal(err)
	}
	copy(h.QueryWeight, []float32{1, 0, 0, 1})
	copy(h.KeyWeight, h.QueryWeight)
	copy(h.ValueWeight, h.QueryWeight)
	context := [][][]float32{{{1, -1}, {-1, 1}, {100, 200}}}
	options := [][][]float32{{{1, -1}, {-1, 1}}}
	cm := [][]bool{{true, true, false}}
	om := [][]bool{{true, true}}
	logits, err := h.Forward(context, cm, options, om)
	if err != nil {
		t.Fatal(err)
	}
	// LayerNorm reduces each real vector to +/-a; symmetric keys yield tanh attention.
	a := 1 / math.Sqrt(1+1e-5)
	score := 2 * a * a / math.Sqrt(2)
	want := score * math.Tanh(score)
	for _, v := range logits[0] {
		if math.Abs(float64(v)-want) > 1e-5 {
			t.Fatalf("logit=%g want %g", v, want)
		}
	}
	options[0][0], options[0][1] = options[0][1], options[0][0]
	swapped, err := h.Forward(context, cm, options, om)
	if err != nil {
		t.Fatal(err)
	}
	if swapped[0][0] != logits[0][1] || swapped[0][1] != logits[0][0] {
		t.Fatal("option permutation changed scoring")
	}
}

func TestTinyParameterRoundTrip(t *testing.T) {
	m, err := NewTinyScorer(Config{Width: 2, Rank: 2, ContextTokens: 8, OptionTokens: 8})
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range m.NamedParameters() {
		for i := range p.Values {
			p.Values[i] = float32(i%7) * 0.1
		}
	}
	batch, err := BuildByteBatch([]ChoiceExample{{Context: "abc", Options: []string{"ab", "cd"}, Label: 0}}, 8, 8)
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.PredictBatch(batch)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(out.Probabilities[0][0]+out.Probabilities[0][1])-1) > 1e-6 {
		t.Fatal("probabilities not normalised")
	}
	other, _ := NewTinyScorer(m.Config)
	if err := other.LoadNamedParameters(m.NamedParameterMap()); err != nil {
		t.Fatal(err)
	}
	got, err := other.Forward(batch)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got[0] {
		if v != out.Logits[0][i] {
			t.Fatal("parameter roundtrip changed logits")
		}
	}
}
