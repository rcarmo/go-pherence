package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strconv"
	"testing"
)

func TestPublishedMixedRecordParity(t *testing.T) {
	dir := os.Getenv("GLINER_MODEL_DIR")
	if dir == "" {
		t.Skip("set GLINER_MODEL_DIR")
	}
	var ref struct {
		Text     string
		IDs      []int
		QueryIDs []int `json:"query_ids"`
		Object   []float32
		Mask     []bool
		Assign   map[string][][]float32
	}
	raw, err := os.ReadFile("testdata/mixed_record_reference.json")
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
	spec := &RecordSpec{Mode: RecordModeNatural, AnchorQueryID: 0, Fields: []RecordField{{QueryID: 0, Name: "name", Scalar: true}, {QueryID: 1, Name: "city", Scalar: true}}}
	schemas := []TextSchema{{Parent: "person", Marker: "[C]", Labels: []string{"name", "city"}, Record: spec}, {Parent: "entities", Marker: "[E]", Labels: []string{"location"}}}
	got, err := m.ScoreSchemas(ref.Text, schemas, 512)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.Input.IDs, ref.IDs) || !reflect.DeepEqual(got.GroupQueryIDs[0], ref.QueryIDs) {
		t.Fatal("routing mismatch")
	}
	g := got.Records[0].Group
	if !reflect.DeepEqual(g.InstanceMask, ref.Mask) {
		t.Fatal("instance masks")
	}
	var delta float64
	for i, v := range g.ObjectLogits {
		if math.Abs(float64(v-ref.Object[i])) > 2e-3 {
			t.Fatal("object parity")
		}
	}
	for key, rows := range ref.Assign {
		i, _ := strconv.Atoi(key)
		for j, row := range rows {
			for k, v := range row {
				d := math.Abs(float64(g.AssignLogits[i][j][k] - v))
				delta = math.Max(delta, d)
				if d > 2e-3 {
					t.Fatal("assignment parity", i, j, k, d)
				}
			}
		}
	}
	if _, err = DecodeSchemaGroups(ref.Text, got, .5, "flat", m.Config.BoundaryHead); err != nil {
		t.Fatal(err)
	}
	t.Logf("mixed record max delta %g", delta)
}
