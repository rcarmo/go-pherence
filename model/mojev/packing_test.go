package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
)

func TestPinnedMoJevTextPacking(t *testing.T) {
	for _, pin := range []struct{ path, hash string }{
		{"../../scripts/mojev_oracle_text_packing.py", "2a88269b2197a3b6d67403ab5759d52510b998c4db64d4a1b6fdee0fc1df2492"},
		{"testdata/text_packing.json", "1a2cb02f7eb9db2fb0b60922e18dc5a8d3224a76465467ca87a465a33b7267a8"},
	} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.hash {
			t.Fatalf("oracle hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/text_packing.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema   int    `json:"schema"`
		Revision string `json:"source_revision"`
		FullSHA  string `json:"full_sha256"`
		PadID    int    `json:"pad_id"`
		Input    []struct {
			State      []int     `json:"state"`
			Questions  [][]int   `json:"questions"`
			Candidates [][][]int `json:"candidates"`
		} `json:"input"`
		Expected struct {
			IDs        [][]int     `json:"packed_ids"`
			PackedMask [][]int     `json:"packed_mask"`
			State      [][]int     `json:"context_span"`
			Questions  [][][]int   `json:"field_span"`
			Candidates [][][][]int `json:"option_span"`
			OptionMask [][][]int   `json:"option_mask"`
		} `json:"expected"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != 1 || fixture.Revision != SourceRevision || fixture.FullSHA != "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79" || len(fixture.Input) != 2 {
		t.Fatal("unexpected fixture provenance")
	}
	rows := make([]EncodedRow, len(fixture.Input))
	for i, row := range fixture.Input {
		rows[i] = EncodedRow{State: row.State, Questions: row.Questions, Candidates: row.Candidates}
	}
	got, err := PackEncodedRows(rows, fixture.PadID)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got.IDs, fixture.Expected.IDs) {
		t.Fatal("token ID packing mismatch")
	}
	check := func(label string, actual any, expected any) {
		t.Helper()
		if !reflect.DeepEqual(actual, expected) {
			t.Fatalf("%s packing mismatch: actual=%v expected=%v", label, actual, expected)
		}
	}
	ints := func(row []bool) []int {
		out := make([]int, len(row))
		for i, v := range row {
			if v {
				out[i] = 1
			}
		}
		return out
	}
	pack2 := func(rows [][]bool) [][]int {
		out := make([][]int, len(rows))
		for i, row := range rows {
			out[i] = ints(row)
		}
		return out
	}
	pack3 := func(rows [][][]bool) [][][]int {
		out := make([][][]int, len(rows))
		for i, row := range rows {
			out[i] = pack2(row)
		}
		return out
	}
	pack4 := func(rows [][][][]bool) [][][][]int {
		out := make([][][][]int, len(rows))
		for i, row := range rows {
			out[i] = pack3(row)
		}
		return out
	}
	check("packed", pack2(got.PackedMask), fixture.Expected.PackedMask)
	check("state", pack2(got.State), fixture.Expected.State)
	check("questions", pack3(got.Questions), fixture.Expected.Questions)
	check("candidates", pack4(got.Candidates), fixture.Expected.Candidates)
	check("option mask", pack3(got.OptionMask), fixture.Expected.OptionMask)
	for r := range rows {
		if !reflect.DeepEqual(rows[r].State, fixture.Input[r].State) || !reflect.DeepEqual(rows[r].Questions, fixture.Input[r].Questions) || !reflect.DeepEqual(rows[r].Candidates, fixture.Input[r].Candidates) {
			t.Fatal("mutated encoded input")
		}
	}
	got.IDs[0][0] = 99
	if rows[0].State[0] == 99 || got.IDs[1][0] == 99 {
		t.Fatal("output aliases input or another row")
	}
	for r := range got.IDs {
		if _, err := AdditiveTreeMask(got.State[r], got.Questions[r], got.Candidates[r], got.PackedMask[r]); err != nil {
			t.Fatalf("packed row %d rejected by tree mask: %v", r, err)
		}
	}
}

func TestPackEncodedRowsRejectsMalformed(t *testing.T) {
	base := EncodedRow{State: []int{1}, Questions: [][]int{{2}}, Candidates: [][][]int{{{3}, {4}}}}
	for name, tc := range map[string]struct {
		rows []EncodedRow
		pad  int
	}{
		"no rows": {nil, 0}, "too many rows": {make([]EncodedRow, 17), 0}, "negative pad": {[]EncodedRow{base}, -1},
		"empty state":         {[]EncodedRow{{Questions: base.Questions, Candidates: base.Candidates}}, 0},
		"no questions":        {[]EncodedRow{{State: base.State}}, 0},
		"missing group":       {[]EncodedRow{{State: base.State, Questions: base.Questions}}, 0},
		"empty question":      {[]EncodedRow{{State: base.State, Questions: [][]int{{}}, Candidates: base.Candidates}}, 0},
		"empty candidate":     {[]EncodedRow{{State: base.State, Questions: base.Questions, Candidates: [][][]int{{{}}}}}, 0},
		"too many candidates": {[]EncodedRow{{State: base.State, Questions: base.Questions, Candidates: [][][]int{make([][]int, 65)}}}, 0},
		"one candidate":       {[]EncodedRow{{State: base.State, Questions: base.Questions, Candidates: [][][]int{{{3}}}}}, 0},
		"mismatched fields":   {[]EncodedRow{base, {State: []int{1}, Questions: [][]int{{2}, {3}}, Candidates: [][][]int{{{4}}, {{5}}}}}, 0},
		"negative state":      {[]EncodedRow{{State: []int{-1}, Questions: base.Questions, Candidates: base.Candidates}}, 0},
		"negative question":   {[]EncodedRow{{State: base.State, Questions: [][]int{{-1}}, Candidates: base.Candidates}}, 0},
		"negative candidate":  {[]EncodedRow{{State: base.State, Questions: base.Questions, Candidates: [][][]int{{{-1}}}}}, 0},
		"too long":            {[]EncodedRow{{State: make([]int, 4096), Questions: base.Questions, Candidates: base.Candidates}}, 0},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := PackEncodedRows(tc.rows, tc.pad)
			if err == nil || got != nil {
				t.Fatalf("accepted malformed rows: %v", got)
			}
		})
	}
}
