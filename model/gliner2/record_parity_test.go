package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestRecordGroupUpstreamParity(t *testing.T) {
	var ref struct {
		Weights         referenceTensors
		Queries, States [][]float32
		PairLogits      [][]float32 `json:"pair_logits"`
		Cases           []struct {
			Mode   string
			Object []float32
			Assign [][][]float32
			Mask   []bool
		}
	}
	raw, err := os.ReadFile("testdata/record_group_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &ref); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	c, err := LoadConfig(f)
	if err != nil {
		t.Fatal(err)
	}
	c.BoundaryHead.RecordDim = 2
	c.BoundaryHead.RecordInstanceQueries = 2
	head, err := LoadRecordHead(ref.Weights, 4, c.BoundaryHead)
	if err != nil {
		t.Fatal(err)
	}
	pool := PooledCandidates{Indices: [][]int{{0, 1}, {1, 2}, {2, 3}}, ValidMask: []bool{true, true, false}}
	for _, tc := range ref.Cases {
		spec := RecordSpec{Mode: tc.Mode, AnchorQueryID: 0, Fields: []RecordField{{QueryID: 0, Name: "name", Scalar: true}, {QueryID: 1, Name: "place", Scalar: true}}}
		got, err := head.ForwardGroupDense(spec, ref.Queries, ref.States, pool, ref.PairLogits, []bool{true, true})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got.InstanceMask, tc.Mask) {
			t.Fatal("mask", tc.Mode)
		}
		for i, v := range got.ObjectLogits {
			if math.Abs(float64(v-tc.Object[i])) > 2e-5 {
				t.Fatal("object", tc.Mode, i)
			}
		}
		for i, row := range got.AssignLogits {
			for j, col := range row {
				for k, v := range col {
					if math.Abs(float64(v-tc.Assign[i][j][k])) > 2e-5 {
						t.Fatalf("%s assign %d,%d,%d got%g want%g", tc.Mode, i, j, k, v, tc.Assign[i][j][k])
					}
				}
			}
		}
	}
}
