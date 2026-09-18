package gliner2

import (
	"math"
	"strings"
	"testing"
)

func TestTypedRelationPairGeneratorTypedThresholdSelfAndCaps(t *testing.T) {
	scores := makeRelationTestScores(
		[]string{"person", "org", "place"},
		[][2]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}, {0, 0}},
		[]bool{true, true, true, true, false},
		[][]float64{
			{0.90, 0.20, 0.10},
			{0.80, 0.70, 0.30},
			{0.60, 0.95, 0.40},
			{0.77, 0.88, 0.92},
			{0.99, 0.99, 0.99},
		},
	)
	generator := TypedRelationPairGenerator{Settings: RelationProposalSettings{
		HeadsPerRelation: 2,
		TailsPerRelation: 2,
		PairCap:          2,
		Threshold:        0.75,
	}}
	schema := []RelationTypeSpec{
		{RelationType: "employment", HeadQueryIDs: []int{0}, TailQueryIDs: []int{1}},
		{RelationType: "peer", HeadQueryIDs: []int{1}, TailQueryIDs: []int{1}},
	}

	got, err := generator.Generate(scores, schema)
	if err != nil {
		t.Fatal(err)
	}
	if got.Len() != 4 || len(got.PairMask) != 4 {
		t.Fatalf("len=%d mask=%d want 4", got.Len(), len(got.PairMask))
	}
	want := []RelationPair{
		{RelationIndex: 0, RelationType: "employment", HeadQueryID: 0, HeadCandidateIndex: 0, HeadStart: 0, HeadEnd: 1, TailQueryID: 1, TailCandidateIndex: 2, TailStart: 2, TailEnd: 3, HeadProbability: 0.90, TailProbability: 0.95, Priority: 1.85},
		{RelationIndex: 0, RelationType: "employment", HeadQueryID: 0, HeadCandidateIndex: 0, HeadStart: 0, HeadEnd: 1, TailQueryID: 1, TailCandidateIndex: 3, TailStart: 3, TailEnd: 4, HeadProbability: 0.90, TailProbability: 0.88, Priority: 1.78},
		{RelationIndex: 1, RelationType: "peer", HeadQueryID: 1, HeadCandidateIndex: 2, HeadStart: 2, HeadEnd: 3, TailQueryID: 1, TailCandidateIndex: 3, TailStart: 3, TailEnd: 4, HeadProbability: 0.95, TailProbability: 0.88, Priority: 1.83},
		{RelationIndex: 1, RelationType: "peer", HeadQueryID: 1, HeadCandidateIndex: 3, HeadStart: 3, HeadEnd: 4, TailQueryID: 1, TailCandidateIndex: 2, TailStart: 2, TailEnd: 3, HeadProbability: 0.88, TailProbability: 0.95, Priority: 1.83},
	}
	for i := range want {
		if !got.PairMask[i] {
			t.Fatalf("slot %d unexpectedly invalid: %+v", i, got.Pairs[i])
		}
		compareRelationPair(t, got.Pairs[i], want[i], 1e-6)
	}
}

func TestTypedRelationPairGeneratorDeterministicTiePolicy(t *testing.T) {
	scores := makeRelationTestScores(
		[]string{"head_a", "head_b"},
		[][2]int{{0, 1}, {1, 2}},
		[]bool{true, true},
		[][]float64{
			{0.90, 0.90},
			{0.90, 0.10},
		},
	)
	generator := TypedRelationPairGenerator{Settings: RelationProposalSettings{
		HeadsPerRelation: 2,
		TailsPerRelation: 1,
		PairCap:          2,
		Threshold:        0.50,
	}}
	got, err := generator.Generate(scores, []RelationTypeSpec{{
		RelationType: "ties",
		HeadQueryIDs: []int{0, 1},
		TailQueryIDs: []int{1},
		AllowSelf:    true,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if !got.PairMask[0] || !got.PairMask[1] {
		t.Fatalf("pair mask=%v", got.PairMask)
	}
	if got.Pairs[0].HeadQueryID != 0 || got.Pairs[0].HeadCandidateIndex != 0 {
		t.Fatalf("first tie pick=%+v want query 0 candidate 0", got.Pairs[0])
	}
	if got.Pairs[1].HeadQueryID != 0 || got.Pairs[1].HeadCandidateIndex != 1 {
		t.Fatalf("second tie pick=%+v want query 0 candidate 1", got.Pairs[1])
	}
}

func TestTypedRelationPairGeneratorRejectsInvalidQueryIDs(t *testing.T) {
	scores := makeRelationTestScores(
		[]string{"person", "org"},
		[][2]int{{0, 1}},
		[]bool{true},
		[][]float64{{0.8, 0.7}},
	)
	generator := TypedRelationPairGenerator{Settings: RelationProposalSettings{HeadsPerRelation: 1, TailsPerRelation: 1, PairCap: 1}}
	for _, schema := range [][]RelationTypeSpec{
		{{RelationType: "bad-head", HeadQueryIDs: []int{2}, TailQueryIDs: []int{1}}},
		{{RelationType: "bad-tail", HeadQueryIDs: []int{0}, TailQueryIDs: []int{-1}}},
	} {
		_, err := generator.Generate(scores, schema)
		if err == nil || !strings.Contains(err.Error(), "outside [0,2)") {
			t.Fatalf("invalid schema %v err=%v", schema, err)
		}
	}
}

func TestTypedRelationPairGeneratorEmptySchema(t *testing.T) {
	scores := makeRelationTestScores(
		[]string{"person"},
		[][2]int{{0, 1}},
		[]bool{true},
		[][]float64{{0.9}},
	)
	got, err := TypedRelationPairGenerator{}.Generate(scores, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got.Len() != 0 || len(got.PairMask) != 0 {
		t.Fatalf("unexpected non-empty output: %+v", got)
	}
}

func makeRelationTestScores(labels []string, spans [][2]int, valid []bool, probabilities [][]float64) EntityScores {
	indices := make([][]int, len(spans))
	logits := make([][]float32, len(spans))
	for i := range spans {
		indices[i] = []int{spans[i][0], spans[i][1]}
		logits[i] = make([]float32, len(probabilities[i]))
		for q, probability := range probabilities[i] {
			logits[i][q] = logit32(probability)
		}
	}
	return EntityScores{
		Input: EntityInput{Labels: append([]string(nil), labels...)},
		Candidates: PooledCandidates{
			Indices:   indices,
			ValidMask: append([]bool(nil), valid...),
		},
		Logits: logits,
	}
}

func logit32(probability float64) float32 {
	return float32(math.Log(probability / (1 - probability)))
}

func compareRelationPair(t *testing.T, got, want RelationPair, tol float64) {
	t.Helper()
	for _, values := range []struct {
		name      string
		got, want int
	}{
		{name: "relation_index", got: got.RelationIndex, want: want.RelationIndex},
		{name: "head_query_id", got: got.HeadQueryID, want: want.HeadQueryID},
		{name: "head_candidate_index", got: got.HeadCandidateIndex, want: want.HeadCandidateIndex},
		{name: "head_start", got: got.HeadStart, want: want.HeadStart},
		{name: "head_end", got: got.HeadEnd, want: want.HeadEnd},
		{name: "tail_query_id", got: got.TailQueryID, want: want.TailQueryID},
		{name: "tail_candidate_index", got: got.TailCandidateIndex, want: want.TailCandidateIndex},
		{name: "tail_start", got: got.TailStart, want: want.TailStart},
		{name: "tail_end", got: got.TailEnd, want: want.TailEnd},
	} {
		if values.got != values.want {
			t.Fatalf("%s=%d want %d pair=%+v", values.name, values.got, values.want, got)
		}
	}
	if got.RelationType != want.RelationType {
		t.Fatalf("relation_type=%q want %q", got.RelationType, want.RelationType)
	}
	for _, values := range []struct {
		name      string
		got, want float64
	}{
		{name: "head_probability", got: got.HeadProbability, want: want.HeadProbability},
		{name: "tail_probability", got: got.TailProbability, want: want.TailProbability},
		{name: "priority", got: got.Priority, want: want.Priority},
	} {
		if math.Abs(values.got-values.want) > tol {
			t.Fatalf("%s=%g want %g pair=%+v", values.name, values.got, values.want, got)
		}
	}
}
