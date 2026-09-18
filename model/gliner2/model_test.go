package gliner2

import (
	"math"
	"os"
	"testing"
)

func TestPublishedEntityScoring(t *testing.T) {
	dir := os.Getenv("GLINER_MODEL_DIR")
	if dir == "" {
		t.Skip("set GLINER_MODEL_DIR to published checkpoint")
	}
	m, err := LoadEntityModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	out, err := m.ScoreEntities("Ada Lovelace lived in London.", []string{"person", "location"}, 512)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Logits) != m.Pool.PoolSize || len(out.NullLogits) != 2 || len(out.CountLogits) != 2 {
		t.Fatal("output shape")
	}
	best := float32(-math.MaxFloat32)
	var bi, bq int
	for i, row := range out.Logits {
		for q, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatal("nonfinite logits")
			}
			if out.Candidates.ValidMask[i] && v > best {
				best = v
				bi = i
				bq = q
			}
		}
	}
	t.Logf("best span=%v label=%s logit=%g null=%v count=%v", out.Candidates.Indices[bi], out.Input.Labels[bq], best, out.NullLogits, out.CountLogits)
}
func TestEntityModelRejectsMissing(t *testing.T) {
	if _, err := LoadEntityModel(t.TempDir()); err == nil {
		t.Fatal("missing checkpoint accepted")
	}
	var m *EntityModel
	if _, err := m.ScoreEntities("text", []string{"label"}, 10); err == nil {
		t.Fatal("nil model accepted")
	}
}
