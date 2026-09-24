package simplejev

import (
	"math"
	"slices"
	"testing"
)

func TestChoiceTieAndStability(t *testing.T) {
	labels := []Label{{TokenID: 9, ID: "first", Logit: 5}, {TokenID: 3, ID: "second", Logit: 5}, {TokenID: 7, ID: "third", Logit: 4}}
	original := slices.Clone(labels)
	id, probs, err := Choice(labels)
	if err != nil {
		t.Fatal(err)
	}
	if id != "first" || probs[0] != probs[1] || probs[1] <= probs[2] || !slices.Equal(labels, original) {
		t.Fatalf("choice=%q probabilities=%v labels mutated=%v", id, probs, !slices.Equal(labels, original))
	}
	var sum float64
	for _, p := range probs {
		sum += float64(p)
	}
	if math.Abs(sum-1) > 1e-6 {
		t.Fatalf("probabilities sum=%v", sum)
	}
}

func TestDistributionExtremeFiniteLogits(t *testing.T) {
	labels := []Label{{TokenID: 1, ID: "min", Logit: -math.MaxFloat32}, {TokenID: 2, ID: "max", Logit: math.MaxFloat32}}
	p, err := Distribution(labels)
	if err != nil || p[0] != 0 || p[1] != 1 {
		t.Fatalf("extreme finite logits: p=%v err=%v", p, err)
	}
	p, err = Distribution([]Label{{TokenID: 1, ID: "one", Logit: -math.MaxFloat32}, {TokenID: 2, ID: "two", Logit: -math.MaxFloat32}})
	if err != nil || p[0] != .5 || p[1] != .5 {
		t.Fatalf("equal extreme logits: p=%v err=%v", p, err)
	}
}

func TestChoiceUsesLogitsWhenProbabilitiesUnderflow(t *testing.T) {
	labels := []Label{{TokenID: 1, ID: "low", Logit: -200}, {TokenID: 2, ID: "middle", Logit: -199}, {TokenID: 3, ID: "high", Logit: 0}}
	id, probs, err := Choice(labels)
	if err != nil || id != "high" || probs[0] != 0 || probs[1] != 0 || probs[2] != 1 {
		t.Fatalf("choice=%q probabilities=%v err=%v", id, probs, err)
	}
	labels[2].Logit = -198
	id, _, err = Choice(labels)
	if err != nil || id != "high" {
		t.Fatalf("choice=%q err=%v", id, err)
	}
}

func TestOrdinalExplicitValues(t *testing.T) {
	labels := []Label{{TokenID: 1, ID: "low", Logit: 0, Value: .01}, {TokenID: 2, ID: "high", Logit: 0, Value: .99}}
	value, probs, err := Ordinal(labels)
	if err != nil || math.Abs(value-.5) > 1e-6 || !slices.Equal(probs, []float32{.5, .5}) {
		t.Fatalf("ordinal=%v probabilities=%v err=%v", value, probs, err)
	}
	labels = []Label{{TokenID: 1, ID: "low", Logit: -100, Value: 1}, {TokenID: 2, ID: "high", Logit: 0, Value: 0}}
	value, probs, err = Ordinal(labels)
	if err != nil || value <= 0 || probs[0] <= 0 || value != float64(probs[0]) {
		t.Fatalf("ordinal subnormal probability: value=%v probabilities=%v err=%v", value, probs, err)
	}
}

func TestScoringRejectsMalformed(t *testing.T) {
	base := []Label{{TokenID: 1, ID: "one", Logit: 0}, {TokenID: 2, ID: "two", Logit: 1}}
	cases := map[string][]Label{
		"empty":           nil,
		"one":             base[:1],
		"many":            make([]Label, 51),
		"duplicate ID":    {{TokenID: 1, ID: "same"}, {TokenID: 2, ID: "same"}},
		"duplicate token": {{TokenID: 1, ID: "one"}, {TokenID: 1, ID: "two"}},
		"negative token":  {{TokenID: -1, ID: "one"}, base[1]},
		"empty ID":        {{TokenID: 1}, base[1]},
		"nan logit":       {{TokenID: 1, ID: "one", Logit: float32(math.NaN())}, base[1]},
		"infinite logit":  {{TokenID: 1, ID: "one", Logit: float32(math.Inf(1))}, base[1]},
	}
	for name, labels := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := Distribution(labels); err == nil {
				t.Fatal("accepted malformed labels")
			}
		})
	}
	for _, value := range []float64{math.NaN(), math.Inf(-1), math.MaxFloat64} {
		labels := slices.Clone(base)
		labels[0].Value, labels[1].Value = value, math.MaxFloat64
		if _, _, err := Ordinal(labels); err == nil {
			t.Fatalf("accepted ordinal value=%v", value)
		}
	}
}
