package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestPublishedRelationScoring(t *testing.T) {
	dir := os.Getenv("GLINER_MODEL_DIR")
	if dir == "" {
		t.Skip("set GLINER_MODEL_DIR")
	}
	m, err := LoadEntityModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	result, err := m.ScoreRelation("Ada Lovelace lived in London.", "lives_in", 512)
	if err != nil {
		t.Fatal(err)
	}
	var ref struct {
		IDs   []int
		Pairs []struct {
			Span  [4]int
			Logit float32
		}
	}
	raw, err := os.ReadFile("testdata/published_relation_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(result.Input.IDs, ref.IDs) {
		t.Fatal("relation token IDs differ")
	}
	expected := make(map[[4]int]float32)
	for _, p := range ref.Pairs {
		expected[p.Span] = p.Logit
	}
	actual := make(map[[4]int]float32)
	for i, p := range result.Pairs.Pairs {
		if result.Pairs.PairMask[i] {
			actual[[4]int{p.HeadStart, p.HeadEnd, p.TailStart, p.TailEnd}] = result.Logits[i]
		}
	}
	if len(actual) != len(expected) {
		t.Fatalf("pair sets differ %d vs %d", len(actual), len(expected))
	}
	var maxDelta float64
	for span, want := range expected {
		got, ok := actual[span]
		if !ok {
			t.Fatal("missing pair", span)
		}
		delta := math.Abs(float64(got - want))
		maxDelta = math.Max(maxDelta, delta)
		if delta > 2e-3 {
			t.Fatalf("%v got %g want %g", span, got, want)
		}
	}
	t.Logf("relation logit max delta %g", maxDelta)
	best := float32(-math.MaxFloat32)
	at := -1
	for i, v := range result.Logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("nonfinite logit")
		}
		if result.Pairs.PairMask[i] && v > best {
			best = v
			at = i
		}
	}
	if at < 0 {
		t.Fatal("no valid pairs")
	}
	t.Logf("top relation %+v logit %g", result.Pairs.Pairs[at], best)
}
func TestNilRelationModel(t *testing.T) {
	var m *EntityModel
	if _, err := m.ScoreRelation("text", "r", 5); err == nil {
		t.Fatal("nil model accepted")
	}
}
