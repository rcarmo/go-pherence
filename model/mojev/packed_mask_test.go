package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"
)

func TestPinnedMoJevPackedMask(t *testing.T) {
	for _, pin := range []struct{ path, sha string }{
		{"../../scripts/mojev_oracle_packed_mask.py", "0a70cd7dfe0e8800cba04ca1a6931287b26f294f9b17ccf8181be1101e829c4e"},
		{"testdata/packed_mask.json", "ffa75bba63f5ab1be78bcf14bc907cdccce16971cb26736d52fc8e08fc73faa0"},
	} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.sha {
			t.Fatalf("oracle hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/packed_mask.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema    int    `json:"schema"`
		Repo      string `json:"source_repo"`
		Revision  string `json:"source_revision"`
		ModelSHA  string `json:"modeling_sha256"`
		FloorBits string `json:"masked_float32_bits"`
		Cases     []struct {
			Name       string    `json:"name"`
			State      []int     `json:"state"`
			Questions  [][]int   `json:"questions"`
			Candidates [][][]int `json:"candidates"`
			Packed     []int     `json:"packed"`
			Allowed    [][]int   `json:"allowed"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Repo != "MoLeMo-Lab/mojev" || fixture.Revision != SourceRevision || fixture.ModelSHA != "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458" || fixture.FloorBits != "ff7fffff" || len(fixture.Cases) != 2 {
		t.Fatal("unexpected oracle provenance")
	}
	bits := func(row []int) []bool {
		result := make([]bool, len(row))
		for i, v := range row {
			if v != 0 && v != 1 {
				t.Fatalf("invalid oracle bit %d", v)
			}
			result[i] = v == 1
		}
		return result
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			state, packed := bits(tc.State), bits(tc.Packed)
			questions := make([][]bool, len(tc.Questions))
			candidates := make([][][]bool, len(tc.Candidates))
			for f, row := range tc.Questions {
				questions[f] = bits(row)
			}
			for f, group := range tc.Candidates {
				candidates[f] = make([][]bool, len(group))
				for n, row := range group {
					candidates[f][n] = bits(row)
				}
			}
			got, err := AdditiveTreeMask(state, questions, candidates, packed)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(tc.Allowed) {
				t.Fatal("wrong row count")
			}
			for row, cols := range tc.Allowed {
				if len(got[row]) != len(cols) {
					t.Fatal("wrong column count")
				}
				for column, want := range cols {
					bits := math.Float32bits(got[row][column])
					if (want == 1 && bits != 0) || (want == 0 && bits != 0xff7fffff) {
						t.Fatalf("mask[%d][%d] bits=%08x want allowed=%d", row, column, bits, want)
					}
				}
			}
			if !reflect.DeepEqual(state, bits(tc.State)) || !reflect.DeepEqual(packed, bits(tc.Packed)) {
				t.Fatal("mutated caller mask")
			}
			for f, q := range questions {
				if !reflect.DeepEqual(q, bits(tc.Questions[f])) {
					t.Fatal("mutated question")
				}
				for n, c := range candidates[f] {
					if !reflect.DeepEqual(c, bits(tc.Candidates[f][n])) {
						t.Fatal("mutated candidate")
					}
				}
			}
		})
	}
}

func TestAdditiveTreeMaskRejectsInvalid(t *testing.T) {
	state := []bool{true, false, false}
	q := [][]bool{{false, true, false}}
	c := [][][]bool{{{false, false, true}}}
	for _, tc := range []struct {
		name       string
		state      []bool
		questions  [][]bool
		candidates [][][]bool
		packed     []bool
	}{
		{"packed length", state, q, c, []bool{true}},
		{"invalid tree", state, [][]bool{{true, false, false}}, c, []bool{true, true, true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := AdditiveTreeMask(tc.state, tc.questions, tc.candidates, tc.packed)
			if err == nil || got != nil {
				t.Fatalf("accepted invalid mask: %v", got)
			}
		})
	}
	got, err := AdditiveTreeMask(state, q, c, []bool{false, false, false})
	if err != nil {
		t.Fatal(err)
	}
	for row, cols := range got {
		for col, value := range cols {
			if (row == col) != (value == 0) {
				t.Fatalf("packed diagonal failed [%d][%d]", row, col)
			}
		}
	}
	got[0][0] = 17
	if !state[0] || !q[0][1] || !c[0][0][2] {
		t.Fatal("returned mask aliases caller")
	}
	if got[1][0] == 17 || cap(got[0]) != len(got[0]) {
		t.Fatal("mask rows must not expose or alias adjacent rows")
	}
}
