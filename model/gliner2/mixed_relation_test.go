package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestPublishedMixedRelationParity(t *testing.T) {
	dir := os.Getenv("GLINER_MODEL_DIR")
	if dir == "" {
		t.Skip("set GLINER_MODEL_DIR")
	}
	var ref struct {
		Text  string
		IDs   []int
		Pairs []struct {
			Span  [4]int
			Logit float32
		}
	}
	raw, err := os.ReadFile("testdata/mixed_relation_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	m, err := LoadEntityModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.ScoreSchemas(ref.Text, []TextSchema{{Parent: "entities", Marker: "[E]", Labels: []string{"person", "location"}}, {Parent: "lives_in", Marker: "[R]", Labels: []string{"head", "tail"}}}, 512)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Input.IDs, ref.IDs) {
		t.Fatal("mixed relation token mismatch")
	}
	r := got.Relations[1]
	actual := map[[4]int]float32{}
	for i, p := range r.Pairs.Pairs {
		if r.Pairs.PairMask[i] {
			actual[[4]int{p.HeadStart, p.HeadEnd, p.TailStart, p.TailEnd}] = r.Logits[i]
		}
	}
	if len(actual) != len(ref.Pairs) {
		t.Fatalf("pair counts %d %d", len(actual), len(ref.Pairs))
	}
	for _, p := range ref.Pairs {
		v, ok := actual[p.Span]
		if !ok || math.Abs(float64(v-p.Logit)) > 2e-3 {
			t.Fatalf("pair %v got%g want%g", p.Span, v, p.Logit)
		}
	}
	decoded, err := DecodeSchemaGroups(ref.Text, got, .5, "flat", m.Config.BoundaryHead)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded[1].Relations) == 0 {
		t.Fatal("no decoded relation")
	}
}
