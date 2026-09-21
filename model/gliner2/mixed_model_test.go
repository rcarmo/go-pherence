package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestPublishedMixedScoring(t *testing.T) {
	dir := os.Getenv("GLINER_MODEL_DIR")
	if dir == "" {
		t.Skip("set GLINER_MODEL_DIR")
	}
	var ref struct {
		Text           string
		IDs            []int
		Indices        [][]int
		Valid          []bool
		Logits         [][]float32
		Classification []float32
	}
	raw, err := os.ReadFile("testdata/mixed_model_reference.json")
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
	got, err := m.ScoreSchemas(ref.Text, []TextSchema{{Parent: "entities", Marker: "[E]", Labels: []string{"person", "location"}}, {Parent: "sentiment", Marker: "[L]", Labels: []string{"positive", "negative"}}}, 512)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Input.IDs, ref.IDs) {
		t.Fatalf("tokens %v want %v", got.Input.IDs, ref.IDs)
	}
	if !reflect.DeepEqual(got.Extraction.Candidates.Indices, ref.Indices) || !reflect.DeepEqual(got.Extraction.Candidates.ValidMask, ref.Valid) {
		t.Fatal("candidate mismatch")
	}
	var delta float64
	for i, row := range got.Extraction.Logits {
		for j, v := range row {
			d := math.Abs(float64(v - ref.Logits[i][j]))
			delta = math.Max(delta, d)
			if d > 2e-3 {
				t.Fatalf("mixed logit %d,%d delta%g", i, j, d)
			}
		}
	}
	for i, v := range got.Classifications[1].Logits {
		if math.Abs(float64(v-ref.Classification[i])) > 2e-3 {
			t.Fatal("classification mismatch")
		}
	}
	if len(got.GroupQueryIDs[1]) != 0 || len(got.GroupQueryIDs[0]) != 2 {
		t.Fatal("query partition")
	}
	t.Logf("mixed extraction max delta %g", delta)
}
