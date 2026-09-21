package gliner2

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"
)

type referenceTensors map[string]struct {
	Shape  []int     `json:"shape"`
	Values []float32 `json:"values"`
}

func (r referenceTensors) GetFloat32(name string) ([]float32, []int, error) {
	v, ok := r[name]
	if !ok {
		return nil, nil, fmt.Errorf("missing %s", name)
	}
	return v.Values, v.Shape, nil
}
func TestDebertaFullTinyTransformersParity(t *testing.T) {
	var f struct {
		Config   DebertaConfig
		IDs      []int `json:"ids"`
		Mask     []bool
		Expected [][]float32
		Weights  referenceTensors
	}
	raw, err := os.ReadFile("testdata/deberta_encoder_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	m, err := LoadDeberta(f.Weights, f.Config)
	if err != nil {
		t.Fatal(err)
	}
	got, err := m.Encode(f.IDs, f.Mask)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		for j, v := range got[i] {
			if math.Abs(float64(v-f.Expected[i][j])) > 2e-5 {
				t.Fatalf("row%d col%d got%g want%g", i, j, v, f.Expected[i][j])
			}
		}
	}
	if _, err = m.Encode([]int{999}, []bool{true}); err == nil {
		t.Fatal("bad token accepted")
	}
	if _, err = m.Encode(f.IDs, nil); err == nil {
		t.Fatal("bad mask accepted")
	}
	delete(f.Weights, "encoder.encoder.layer.0.attention.self.query_proj.weight")
	if _, err = LoadDeberta(f.Weights, f.Config); err == nil {
		t.Fatal("missing weight accepted")
	}
}
