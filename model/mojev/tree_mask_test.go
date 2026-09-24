package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
)

func TestPinnedMoJevTreeMask(t *testing.T) {
	const scriptSHA = "4a7ebcdf8733112386766b34348cb6d2d985d210d6e140649823c0fe5866f4e9"
	const fixtureSHA = "200c5864dbf67c941515e195c85a00eb2815ce96cafd34eac93bf340bd5ec57d"
	for _, v := range []struct{ path, hash string }{{"../../scripts/mojev_oracle_tree_mask.py", scriptSHA}, {"testdata/tree_mask.json", fixtureSHA}} {
		data, err := os.ReadFile(v.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != v.hash {
			t.Fatalf("fixture hash changed: %s", v.path)
		}
	}
	data, err := os.ReadFile("testdata/tree_mask.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema      int    `json:"schema"`
		Repo        string `json:"source_repo"`
		Revision    string `json:"source_revision"`
		ModelingSHA string `json:"modeling_sha256"`
		Cases       []struct {
			Name       string    `json:"name"`
			State      []int     `json:"state"`
			Questions  [][]int   `json:"questions"`
			Candidates [][][]int `json:"candidates"`
			Allowed    [][]int   `json:"allowed"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Repo != "MoLeMo-Lab/mojev" || fixture.Revision != SourceRevision || fixture.ModelingSHA != "a8e93f62d92c6748c5d001fef4f9516d6a74b10158d7265f53bab13f1091d458" || len(fixture.Cases) != 2 {
		t.Fatal("unexpected pinned mask provenance")
	}
	toBools := func(row []int) []bool {
		result := make([]bool, len(row))
		for i, v := range row {
			if v != 0 && v != 1 {
				t.Fatalf("invalid fixture bit %d", v)
			}
			result[i] = v == 1
		}
		return result
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			state := toBools(tc.State)
			questions := make([][]bool, len(tc.Questions))
			candidates := make([][][]bool, len(tc.Candidates))
			for f, row := range tc.Questions {
				questions[f] = toBools(row)
			}
			for f, group := range tc.Candidates {
				candidates[f] = make([][]bool, len(group))
				for n, row := range group {
					candidates[f][n] = toBools(row)
				}
			}
			got, err := TreeMask(state, questions, candidates)
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
				for col, want := range cols {
					if got[row][col] != (want == 1) {
						t.Fatalf("mask[%d][%d]=%v want=%d", row, col, got[row][col], want)
					}
				}
			}
			if !reflect.DeepEqual(state, toBools(tc.State)) {
				t.Fatal("mutated state span")
			}
			for f, q := range questions {
				if !reflect.DeepEqual(q, toBools(tc.Questions[f])) {
					t.Fatal("mutated question span")
				}
				for n, c := range candidates[f] {
					if !reflect.DeepEqual(c, toBools(tc.Candidates[f][n])) {
						t.Fatal("mutated candidate span")
					}
				}
			}
		})
	}
}

func TestTreeMaskMultiTokenAndEmptyCandidate(t *testing.T) {
	// State 0:2; first question 2:4; first candidate 4:6;
	// second candidate absent; second question 6:8; last position padding.
	state := []bool{true, true, false, false, false, false, false, false, false}
	questions := [][]bool{
		{false, false, true, true, false, false, false, false, false},
		{false, false, false, false, false, false, true, true, false},
	}
	candidates := [][][]bool{
		{{false, false, false, false, true, true, false, false, false}, make([]bool, 9)},
		{{false, false, false, false, false, false, false, false, false}},
	}
	got, err := TreeMask(state, questions, candidates)
	if err != nil {
		t.Fatal(err)
	}
	for row, want := range map[int][]bool{
		0: {true, true, false, false, false, false, false, false, false},
		2: {true, true, true, true, false, false, false, false, false},
		4: {true, true, true, true, true, true, false, false, false},
		6: {true, true, false, false, false, false, true, true, false},
		8: {false, false, false, false, false, false, false, false, true},
	} {
		if !reflect.DeepEqual(got[row], want) {
			t.Fatalf("row %d = %v, want %v", row, got[row], want)
		}
	}
}

func TestTreeMaskRejectsMalformed(t *testing.T) {
	baseState := []bool{true, false, false, false}
	baseQ := [][]bool{{false, true, false, false}}
	baseC := [][][]bool{{{false, false, true, false}}}
	if _, err := TreeMask(baseState, baseQ, baseC); err != nil {
		t.Fatal(err)
	}
	for name, v := range map[string]struct {
		state      []bool
		questions  [][]bool
		candidates [][][]bool
	}{
		"empty":                  {nil, baseQ, baseC},
		"too long":               {make([]bool, 4097), baseQ, baseC},
		"no question":            {baseState, nil, baseC},
		"group mismatch":         {baseState, baseQ, nil},
		"wrong question width":   {baseState, [][]bool{{true}}, baseC},
		"wrong candidate width":  {baseState, baseQ, [][][]bool{{{true}}}},
		"no candidates":          {baseState, baseQ, [][][]bool{{}}},
		"too many candidates":    {baseState, baseQ, [][][]bool{make([][]bool, 65)}},
		"too many questions":     {baseState, make([][]bool, 257), make([][][]bool, 257)},
		"state overlap":          {[]bool{true, true, false, false}, baseQ, baseC},
		"question overlap":       {baseState, [][]bool{{false, true, true, false}}, baseC},
		"candidate overlap":      {baseState, baseQ, [][][]bool{{{false, true, false, false}}}},
		"sibling overlap":        {baseState, baseQ, [][][]bool{{{false, false, true, false}, {false, false, true, false}}}},
		"cross-question overlap": {baseState, [][]bool{{false, true, false, false}, {false, false, false, true}}, [][][]bool{{{false, false, true, false}}, {{false, false, true, false}}}},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := TreeMask(v.state, v.questions, v.candidates)
			if err == nil || got != nil {
				t.Fatalf("accepted malformed mask %+v", got)
			}
		})
	}
	if _, err := TreeMask([]bool{true, false, false}, [][]bool{{false, true, false}}, [][][]bool{{{false, false, true}}}); err != nil {
		t.Fatal(err)
	}
	got, err := TreeMask([]bool{true, false, false, false}, [][]bool{{false, true, false, false}}, [][][]bool{{{false, false, true, false}}})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got[3], []bool{false, false, false, true}) {
		t.Fatalf("padding row must see only itself: %v", got[3])
	}
	state := []bool{true, false, false}
	question := [][]bool{{false, true, false}}
	candidate := [][][]bool{{{false, false, true}}}
	got, err = TreeMask(state, question, candidate)
	if err != nil {
		t.Fatal(err)
	}
	got[0][0] = false // the returned matrix must not alias any input span
	if !state[0] || !question[0][1] || !candidate[0][0][2] {
		t.Fatal("returned matrix aliases caller spans")
	}
}
