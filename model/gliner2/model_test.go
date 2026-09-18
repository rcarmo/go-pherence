package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
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
	var ref struct {
		IDs         []int
		Indices     [][]int
		Valid       []bool
		Logits      [][]float32
		Null, Count []float32
	}
	raw, err := os.ReadFile("testdata/published_entities_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out.Input.IDs, ref.IDs) {
		t.Fatalf("token routing differs: got %v want %v", out.Input.IDs, ref.IDs)
	}
	if !reflect.DeepEqual(out.Candidates.Indices, ref.Indices) || !reflect.DeepEqual(out.Candidates.ValidMask, ref.Valid) {
		t.Fatal("candidate pool differs from upstream")
	}
	var maxDelta float64
	for i, row := range out.Logits {
		for q, v := range row {
			delta := math.Abs(float64(v - ref.Logits[i][q]))
			maxDelta = math.Max(maxDelta, delta)
			if delta > 2e-3 {
				t.Fatalf("candidate%d query%d got%g want%g", i, q, v, ref.Logits[i][q])
			}
		}
	}
	for q := range ref.Null {
		if math.Abs(float64(out.NullLogits[q]-ref.Null[q])) > 2e-3 || math.Abs(float64(out.CountLogits[q]-ref.Count[q])) > 2e-3 {
			t.Fatal("optional heads differ")
		}
	}
	t.Logf("upstream candidate logits max absolute delta=%g", maxDelta)
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
