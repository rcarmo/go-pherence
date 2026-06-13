package diffusiongemma

import (
	"math"
	"testing"
)

func TestSelfConditioningSoftEmbeddingMatchesRawLogitSoftmaxMatmul(t *testing.T) {
	logits := []float32{1.25, -0.5, 0.75}
	emb := [][]float32{
		{1, 2},
		{-3, 4},
		{5, -6},
	}
	got := make([]float32, 2)
	scratch := make([]float32, 2)
	err := buildSelfConditioningSoftEmbeddingRow(got, logits, 3, 2, 1, scratch, func(id int, dst []float32) error {
		copy(dst, emb[id])
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}

	maxLogit := float64(logits[0])
	for _, v := range logits[1:] {
		if float64(v) > maxLogit {
			maxLogit = float64(v)
		}
	}
	var denom float64
	probs := make([]float64, len(logits))
	for i, v := range logits {
		p := math.Exp(float64(v) - maxLogit)
		probs[i] = p
		denom += p
	}
	want := []float64{0, 0}
	for i, p := range probs {
		p /= denom
		want[0] += p * float64(emb[i][0])
		want[1] += p * float64(emb[i][1])
	}
	for i := range got {
		if math.Abs(float64(got[i])-want[i]) > 1e-6 {
			t.Fatalf("soft embedding[%d] = %.8f, want %.8f (all got=%v want=%v)", i, got[i], want[i], got, want)
		}
	}
}

func TestSelfConditioningSoftEmbeddingIgnoresNegativeInfAndNaN(t *testing.T) {
	logits := []float32{float32(math.Inf(-1)), float32(math.NaN()), 3}
	emb := [][]float32{{99, 99}, {88, 88}, {7, -5}}
	got := make([]float32, 2)
	scratch := make([]float32, 2)
	err := buildSelfConditioningSoftEmbeddingRow(got, logits, 3, 2, 1, scratch, func(id int, dst []float32) error {
		copy(dst, emb[id])
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 7 || got[1] != -5 {
		t.Fatalf("got %v, want sole finite embedding", got)
	}
}

func TestSelfConditioningSoftEmbeddingAppliesTemperatureInverse(t *testing.T) {
	logits := []float32{1, 2}
	emb := [][]float32{{0}, {10}}
	got := make([]float32, 1)
	scratch := make([]float32, 1)
	if err := buildSelfConditioningSoftEmbeddingRow(got, logits, 2, 1, 2, scratch, func(id int, dst []float32) error {
		copy(dst, emb[id])
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	p1 := math.Exp(4) / (math.Exp(2) + math.Exp(4))
	want := 10 * p1
	if math.Abs(float64(got[0])-want) > 1e-6 {
		t.Fatalf("got %.8f want %.8f", got[0], want)
	}
}

func TestSelfConditioningSoftEmbeddingRowsF32MatchesRowHelper(t *testing.T) {
	logits := [][]float32{{1, 2, -1}, {0.5, -0.25, 1.5}}
	emb := []float32{
		1, 2,
		-3, 4,
		5, -6,
	}
	got := make([]float32, 4)
	if err := buildSelfConditioningSoftEmbeddingRowsF32(got, logits, emb, 2, 3, 2, 1.25); err != nil {
		t.Fatal(err)
	}
	want := make([]float32, 4)
	scratch := make([]float32, 2)
	for pos := range logits {
		if err := buildSelfConditioningSoftEmbeddingRow(want[pos*2:(pos+1)*2], logits[pos], 3, 2, 1.25, scratch, func(id int, dst []float32) error {
			copy(dst, emb[id*2:(id+1)*2])
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("got[%d]=%.8f want %.8f all got=%v want=%v", i, got[i], want[i], got, want)
		}
	}
}
