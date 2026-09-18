package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestSharedScorerPublishedBindingsAndMarginals(t *testing.T) {
	f, err := os.Open("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c, err := LoadConfig(f)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/shared_scorer_shapes.json")
	if err != nil {
		t.Fatal(err)
	}
	var src shapeSource
	if err = json.Unmarshal(raw, &src); err != nil {
		t.Fatal(err)
	}
	s, err := LoadSharedPoolScorer(src, 768, c.BoundaryHead)
	if err != nil {
		t.Fatal(err)
	}
	// Zero weights isolate marginal addition and mean-restored inside evidence.
	boundary := make([][]float32, 3)
	for i := range boundary {
		boundary[i] = make([]float32, 128)
	}
	queries := [][]float32{make([]float32, 768), make([]float32, 768)}
	text := [][]float32{make([]float32, 768), make([]float32, 768)}
	pool := PooledCandidates{Indices: [][]int{{0, 2}, {0, 0}}, ValidMask: []bool{true, false}, ProposalLogits: []float32{99, 99}, CompatLogits: []float32{7, 7}}
	m := BoundaryMarginals{StartLogits: [][]float32{{1, 2, 3}, {4, 5, 6}}, EndLogits: [][]float32{{10, 20, 30}, {40, 50, 60}}, InsidePrefix: [][]float32{{0, -1, 0}, {0, 0, 0}}, InsidePrefixMean: []float32{2, 0}}
	scores, states, err := s.Forward(boundary, queries, []bool{true, false}, pool, m, 2, text, []bool{true, true})
	if err != nil {
		t.Fatal(err)
	}
	want := float32(31 + 4/math.Sqrt(2))
	if math.Abs(float64(scores[0][0]-want)) > 1e-5 {
		t.Fatalf("score %g want %g (prior/marginal double count?)", scores[0][0], want)
	}
	if scores[0][1] != MaskLogit || scores[1][0] != MaskLogit {
		t.Fatal("mask leaked")
	}
	for _, v := range states[1] {
		if v != 0 {
			t.Fatal("invalid candidate state leaked")
		}
	}
	delete(src, "boundary_head.shared_pool_scorer.film.weight")
	if _, err := LoadSharedPoolScorer(src, 768, c.BoundaryHead); err == nil {
		t.Fatal("missing weight accepted")
	}
	s.QueryAttentionLayers = 1
	if s.Validate() == nil {
		t.Fatal("unsupported query attention accepted")
	}
}
