package gliner2

import "testing"

func TestMixedDecodePartitionsQueries(t *testing.T) {
	text := "Ada London"
	words, _ := SplitWords(text)
	groups := []SchemaGroupRouting{{Schema: TextSchema{Parent: "places", Marker: "[E]", Labels: []string{"location"}}}, {Schema: TextSchema{Parent: "sentiment", Marker: "[L]", Labels: []string{"positive"}}}, {Schema: TextSchema{Parent: "people", Marker: "[E]", Labels: []string{"person"}}}}
	s := MixedScores{Input: MixedInput{Words: words, Groups: groups}, GroupQueryIDs: [][]int{{1}, nil, {0}}, Classifications: map[int]ClassificationScores{1: {Task: "sentiment", Labels: []string{"positive"}, Logits: []float32{3}, Probabilities: []float64{.95}}}, Extraction: &EntityScores{Input: EntityInput{Words: words, Labels: []string{"person", "location"}}, Candidates: PooledCandidates{Indices: [][]int{{0, 1}, {1, 2}}, ValidMask: []bool{true, true}}, Logits: [][]float32{{5, -5}, {-5, 5}}, NullLogits: []float32{-5, -5}}}
	c := BoundaryHeadConfig{PairTemperature: 1, EnableAbstention: true, AbstentionThreshold: .5}
	got, err := DecodeSchemaGroups(text, s, .5, "flat", c)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Entities[0].Text != "London" || got[2].Entities[0].Text != "Ada" || got[1].Classification == nil {
		t.Fatal(got)
	}
	s.GroupQueryIDs[0][0] = 99
	if _, err = DecodeSchemaGroups(text, s, .5, "flat", c); err == nil {
		t.Fatal("bad query accepted")
	}
	s.GroupQueryIDs[0][0] = 1
	s.Input.Groups[0].Schema.Marker = "[R]"
	if _, err = DecodeSchemaGroups(text, s, .5, "flat", c); err == nil {
		t.Fatal("relation decoded as entity")
	}
}

func TestMixedClassificationDecodeOwnsSlices(t *testing.T) {
	schema := TextSchema{Parent: "sentiment", Marker: "[L]", Labels: []string{"positive"}}
	s := MixedScores{Input: MixedInput{Groups: []SchemaGroupRouting{{Schema: schema}}}, GroupQueryIDs: [][]int{nil}, Classifications: map[int]ClassificationScores{0: {Task: "sentiment", Labels: []string{"positive"}, Logits: []float32{3}, Probabilities: []float64{.95}}}}
	got, err := DecodeSchemaGroups("text", s, .5, "flat", BoundaryHeadConfig{})
	if err != nil {
		t.Fatal(err)
	}
	got[0].Classification.Labels[0] = "changed"
	got[0].Classification.Logits[0] = 0
	got[0].Classification.Probabilities[0] = 0
	before := s.Classifications[0]
	if before.Labels[0] != "positive" || before.Logits[0] != 3 || before.Probabilities[0] != .95 {
		t.Fatal("decode result aliases raw scores")
	}
}
