package gliner2

import (
	"math"
	"os"
	"testing"
)

func TestPublishedRecordScores(t *testing.T) {
	dir := os.Getenv("GLINER_MODEL_DIR")
	if dir == "" {
		t.Skip("set GLINER_MODEL_DIR")
	}
	m, err := LoadRecordModel(dir)
	if err != nil {
		t.Fatal(err)
	}
	spec := RecordSpec{Mode: RecordModeNatural, AnchorQueryID: 0, Fields: []RecordField{{QueryID: 0, Name: "name", Scalar: true}, {QueryID: 1, Name: "city", Scalar: true}}}
	got, err := m.ScoreRecord("Ada Lovelace lived in London.", "person", spec, 512)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Group.ObjectLogits) != m.Base.Pool.PoolSize {
		t.Fatal("instance count")
	}
	best := float32(-math.MaxFloat32)
	at := -1
	for i, v := range got.Group.ObjectLogits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatal("nonfinite")
		}
		if got.Group.InstanceMask[i] && v > best {
			best = v
			at = i
		}
	}
	if at < 0 {
		t.Fatal("no instances")
	}
	t.Logf("top record anchor=%v logit=%g", got.Group.PoolSpans[at], best)
}
