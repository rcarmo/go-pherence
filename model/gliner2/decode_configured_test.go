package gliner2

import (
	"math"
	"testing"
)

func TestConfiguredDecodeCountUnionAndAbstention(t *testing.T) {
	text := "aa bb cc"
	s := makeDecodeScores(t, text, []string{"item"}, []decodeRow{{0, 1, []float64{.9}}, {1, 2, []float64{.4}}, {2, 3, []float64{.2}}})
	s.CountLogits = []float32{float32(math.Log(2))}
	s.NullLogits = []float32{0}
	c := BoundaryHeadConfig{PairTemperature: 1, EnableAbstention: true, AbstentionThreshold: .5, AdaptiveThreshold: true}
	got, err := DecodeConfiguredEntities(text, s, .5, "allow", c)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatal(got)
	}
	// Strict > threshold means a null probability exactly .5 does not abstain.
	s.NullLogits[0] = 1
	got, err = DecodeConfiguredEntities(text, s, .5, "allow", c)
	if err != nil || len(got) != 0 {
		t.Fatal(got, err)
	}
	s.NullLogits[0] = -1
	s.CountLogits[0] = -100
	got, err = DecodeConfiguredEntities(text, s, .5, "allow", c)
	if err != nil || len(got) != 1 {
		t.Fatal("count removed threshold hit", got, err)
	}
	c.PairTemperature = 0
	if _, err = DecodeConfiguredEntities(text, s, .5, "allow", c); err == nil {
		t.Fatal("zero temperature accepted")
	}
}
