package gliner2

import (
	"math"
	"os"
	"strings"
	"testing"
)

func TestSparseRelationScorerPublishedBindings(t *testing.T) {
	f, err := os.Open("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatal(err)
	}
	src := shapeSource{
		"relation_scorer.mlp.0.weight":                   {768, 4610},
		"relation_scorer.mlp.0.bias":                     {768},
		"relation_scorer.mlp.3.weight":                   {1, 768},
		"relation_scorer.mlp.3.bias":                     {1},
		"relation_scorer.head_content_projection.weight": {768, 768},
		"relation_scorer.head_content_projection.bias":   {768},
		"relation_scorer.tail_content_projection.weight": {768, 768},
		"relation_scorer.tail_content_projection.bias":   {768},
		"relation_scorer.relation_content_gate.weight":   {768, 1536},
		"relation_scorer.relation_content_gate.bias":     {768},
		"relation_scorer.content_linear.weight":          {1, 3072},
		"relation_scorer.content_linear.bias":            {1},
	}
	scorer, err := LoadSparseRelationScorer(src, 768, cfg.BoundaryHead)
	if err != nil {
		t.Fatal(err)
	}
	if scorer.RelationQueryDim != 1536 || !scorer.UseBiaffineContent {
		t.Fatalf("relation scorer binding mismatch: %+v", scorer)
	}
	if scorer.MLPInput.InDim != 4610 || scorer.MLPOutput.InDim != 768 {
		t.Fatalf("unexpected mlp dims in=%d out.in=%d", scorer.MLPInput.InDim, scorer.MLPOutput.InDim)
	}
	delete(src, "relation_scorer.mlp.0.weight")
	if _, err := LoadSparseRelationScorer(src, 768, cfg.BoundaryHead); err == nil {
		t.Fatal("missing published tensor accepted")
	}

	cfg.BoundaryHead.DirectionalRelationStates = false
	cfg.BoundaryHead.RelationBiaffineContent = false
	src = shapeSource{
		"relation_scorer.mlp.0.weight": {768, 3842},
		"relation_scorer.mlp.0.bias":   {768},
		"relation_scorer.mlp.3.weight": {1, 768},
		"relation_scorer.mlp.3.bias":   {1},
	}
	scorer, err = LoadSparseRelationScorer(src, 768, cfg.BoundaryHead)
	if err != nil {
		t.Fatal(err)
	}
	if scorer.RelationQueryDim != 768 || scorer.UseBiaffineContent || scorer.HeadContentProjection != nil {
		t.Fatalf("directional/content toggle mismatch: %+v", scorer)
	}
}

func TestSparseRelationScorerScalarContentMaskingAndInvalidRelation(t *testing.T) {
	scorer := SparseRelationScorer{
		HiddenSize:         1,
		RelationQueryDim:   1,
		UseBiaffineContent: true,
		MLPInput: Linear{
			InDim:  7,
			OutDim: 1,
			Weight: []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7},
			Bias:   []float32{0.05},
		},
		MLPOutput: Linear{
			InDim:  1,
			OutDim: 1,
			Weight: []float32{1.25},
			Bias:   []float32{-0.1},
		},
		HeadContentProjection: &Linear{InDim: 1, OutDim: 1, Weight: []float32{2}, Bias: []float32{0.1}},
		TailContentProjection: &Linear{InDim: 1, OutDim: 1, Weight: []float32{3}, Bias: []float32{-0.2}},
		RelationContentGate:   &Linear{InDim: 1, OutDim: 1, Weight: []float32{2}, Bias: []float32{-1}},
		ContentLinear:         &Linear{InDim: 3, OutDim: 1, Weight: []float32{0.1, 0.2, 0.3}, Bias: []float32{0.4}},
	}
	boundary := [][]float32{{1}, {2}, {4}, {8}}
	relations := [][]float32{{0.5}}
	pairs := RelationPairProposals{
		Pairs: []RelationPair{
			{RelationIndex: 0, HeadStart: 1, HeadEnd: 3, TailStart: 0, TailEnd: 1},
			{RelationIndex: 0, HeadStart: 1, HeadEnd: 3, TailStart: 0, TailEnd: 1},
			{RelationIndex: 7, HeadStart: 1, HeadEnd: 3, TailStart: 0, TailEnd: 1},
		},
		PairMask: []bool{true, false, true},
	}
	got, err := scorer.Forward(boundary, relations, pairs)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 {
		t.Fatalf("scores=%d want 3", len(got))
	}
	pre := float32(0.05 + 0.1*2 + 0.2*4 + 0.3*1 + 0.4*1 + 0.5*0.5 + 0.6*(-1) + 0.7*0.25)
	mlp := 1.25*gelu32(pre) - 0.1
	headContent := float32(2*3 + 0.1)
	tailContent := float32(3*1 - 0.2)
	gate := Sigmoid(2*0.5 - 1)
	biaffine := headContent * gate * tailContent
	content := float32(0.1*headContent + 0.2*tailContent + 0.3*0.5 + 0.4)
	want := mlp + biaffine + content
	if math.Abs(float64(got[0]-want)) > 1e-5 {
		t.Fatalf("score=%g want %g", got[0], want)
	}
	if got[1] != 0 || got[2] != 0 {
		t.Fatalf("masked/invalid outputs leaked: %v", got)
	}
}

func TestSparseRelationScorerShapeValidation(t *testing.T) {
	scorer := SparseRelationScorer{
		HiddenSize:       2,
		RelationQueryDim: 4,
		MLPInput:         Linear{InDim: 14, OutDim: 2, Weight: make([]float32, 28), Bias: make([]float32, 2)},
		MLPOutput:        Linear{InDim: 2, OutDim: 1, Weight: make([]float32, 2), Bias: []float32{0}},
	}
	pair := RelationPairProposals{Pairs: []RelationPair{{RelationIndex: 0}}}
	for _, tc := range []struct {
		name      string
		boundary  [][]float32
		relations [][]float32
		pairs     RelationPairProposals
		needle    string
	}{
		{
			name:      "pair mask length",
			boundary:  [][]float32{{0, 0}},
			relations: [][]float32{{0, 0, 0, 0}},
			pairs:     RelationPairProposals{Pairs: pair.Pairs, PairMask: []bool{true, false}},
			needle:    "pair_mask len=2 want=1",
		},
		{
			name:      "boundary width",
			boundary:  [][]float32{{0, 0}, {1}},
			relations: [][]float32{{0, 0, 0, 0}},
			pairs:     pair,
			needle:    "boundary_states[1] len=1 want=2",
		},
		{
			name:      "relation width",
			boundary:  [][]float32{{0, 0}},
			relations: [][]float32{{0, 0, 0}},
			pairs:     pair,
			needle:    "relation_query_states[0] len=3 want=4",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := scorer.Forward(tc.boundary, tc.relations, tc.pairs)
			if err == nil || !strings.Contains(err.Error(), tc.needle) {
				t.Fatalf("err=%v want substring %q", err, tc.needle)
			}
		})
	}
}
