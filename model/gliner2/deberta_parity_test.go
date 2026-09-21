package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestDebertaTransformersAttentionParity(t *testing.T) {
	var f struct {
		Heads             int    `json:"heads"`
		Buckets           int    `json:"buckets"`
		MaxPosition       int    `json:"max_position"`
		Mask              []bool `json:"mask"`
		Query, Key, Value [][]float32
		RelativeQuery     [][]float32 `json:"relative_query"`
		RelativeKey       [][]float32 `json:"relative_key"`
		Expected          [][]float32 `json:"expected"`
	}
	data, err := os.ReadFile("testdata/deberta_attention_reference.json")
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	got, err := DisentangledAttention(f.Query, f.Key, f.Value, f.RelativeQuery, f.RelativeKey, f.Mask, f.Heads, f.Buckets, f.MaxPosition)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		for j, v := range got[i] {
			if math.Abs(float64(v-f.Expected[i][j])) > 2e-6 {
				t.Fatalf("row%d dim%d got%g want%g", i, j, v, f.Expected[i][j])
			}
		}
	}
}
