package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strconv"
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
	var ref struct {
		IDs    []int
		Spans  [][]int
		Mask   []bool
		Object []float32
		Assign map[string][][]float32
	}
	raw, err := os.ReadFile("testdata/published_record_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Input.IDs, ref.IDs) || !reflect.DeepEqual(got.Group.PoolSpans, ref.Spans) || !reflect.DeepEqual(got.Group.InstanceMask, ref.Mask) {
		t.Fatal("record routing/pool/mask differs")
	}
	var delta float64
	for i, v := range got.Group.ObjectLogits {
		if math.Abs(float64(v-ref.Object[i])) > 2e-3 {
			t.Fatal("object logit differs", i)
		}
	}
	for key, rows := range ref.Assign {
		i, err := strconv.Atoi(key)
		if err != nil {
			t.Fatal(err)
		}
		for j, row := range rows {
			for k, want := range row {
				d := math.Abs(float64(got.Group.AssignLogits[i][j][k] - want))
				delta = math.Max(delta, d)
				if d > 2e-3 {
					t.Fatalf("assignment %d,%d,%d delta%g", i, j, k, d)
				}
			}
		}
	}
	t.Logf("published record max assignment delta=%g", delta)
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
