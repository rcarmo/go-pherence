package gliner2

import (
	"math"
	"os"
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
