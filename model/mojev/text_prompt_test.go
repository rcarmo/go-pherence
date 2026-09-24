package mojev

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestReleasedMoJevTextPacking(t *testing.T) {
	for _, pin := range []struct{ path, hash string }{
		{"../../scripts/mojev_oracle_released_packing.py", "78adb5f49fd61fc63f6654d3ff2bc1da697d1ba1eeafbb76ee7d8a6bd660f236"},
		{"testdata/released_packing.json", "e7501c7c2248d6840fa184b8498e7b5d860c629c6304265a3e8ea74e5b6f6408"},
	} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.hash {
			t.Fatalf("oracle hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/released_packing.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema        int    `json:"schema"`
		Revision      string `json:"source_revision"`
		SchemaSHA     string `json:"schema_sha256"`
		FullSHA       string `json:"full_sha256"`
		TokenSHA      string `json:"tokenizer_sha256"`
		ConfigSHA     string `json:"tokenizer_config_sha256"`
		StateLimit    int    `json:"state_limit"`
		QuestionLimit int    `json:"question_limit"`
		PadID         int    `json:"pad_id"`
		Fields        []struct {
			Name        string   `json:"name"`
			Kind        string   `json:"kind"`
			Options     []string `json:"options"`
			Description string   `json:"description"`
		} `json:"fields"`
		Rows []struct {
			Context string     `json:"context"`
			Menus   [][]string `json:"menus"`
		} `json:"rows"`
		Prompts  []string `json:"prompts"`
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
	if fixture.Schema != 1 || fixture.Revision != SourceRevision || fixture.SchemaSHA != "63562314b7e6c1ed7e7629497ccc0022771e876fac7730f5ce7d038b50be0df7" || fixture.FullSHA != "5e2972605c190a511cdfb3a1c01358c4855b229c93a1679a6c569c5b43b07b79" || fixture.TokenSHA != "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523" || fixture.ConfigSHA != "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b" || len(fixture.Fields) != 2 || len(fixture.Rows) != 2 {
		t.Fatal("unexpected oracle provenance")
	}
	fields := make([]TextField, len(fixture.Fields))
	for f, item := range fixture.Fields {
		if item.Kind != "choice" {
			t.Fatal("unexpected field kind")
		}
		fields[f] = TextField{item.Name, item.Description, item.Options}
		prompt, err := renderChoicePrompt(fields[f])
		if err != nil || prompt != fixture.Prompts[f] {
			t.Fatalf("prompt mismatch: %q / %v", prompt, err)
		}
	}
	rows := make([]TextRow, len(fixture.Rows))
	for r, item := range fixture.Rows {
		rows[r] = TextRow{item.Context, item.Menus}
	}
	path := os.Getenv("GO_PHERENCE_MOJEV_CHECKPOINT_DIR")
	if path == "" {
		t.Skip("set GO_PHERENCE_MOJEV_CHECKPOINT_DIR to the approved tokenizer assets")
	}
	for _, pin := range []struct{ name, hash string }{{"tokenizer.json", fixture.TokenSHA}, {"tokenizer_config.json", fixture.ConfigSHA}} {
		file, err := os.Open(filepath.Join(path, pin.name))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.New()
		_, err = io.Copy(digest, file)
		closeErr := file.Close()
		if err != nil {
			t.Fatal(err)
		}
		if closeErr != nil {
			t.Fatal(closeErr)
		}
		if fmt.Sprintf("%x", digest.Sum(nil)) != pin.hash {
			t.Fatalf("%s hash mismatch", pin.name)
		}
	}
	tok, err := tokenizer.LoadWithConfig(path)
	if err != nil {
		t.Fatal(err)
	}
	got, err := PackTextRows(rows, fields, fixture.StateLimit, fixture.QuestionLimit, fixture.PadID, func(text string) ([]int, error) { return tok.Encode(text), nil })
	if err != nil {
		t.Fatal(err)
	}
	check := func(label string, a, b any) {
		t.Helper()
		if !reflect.DeepEqual(a, b) {
			t.Fatalf("%s mismatch: got %v want %v", label, a, b)
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
	check("IDs", got.IDs, fixture.Expected.IDs)
	check("packed", pack2(got.PackedMask), fixture.Expected.PackedMask)
	check("state", pack2(got.State), fixture.Expected.State)
	check("questions", pack3(got.Questions), fixture.Expected.Questions)
	check("candidates", pack4(got.Candidates), fixture.Expected.Candidates)
	check("option mask", pack3(got.OptionMask), fixture.Expected.OptionMask)
}

func TestPackTextRowsRejectsMalformed(t *testing.T) {
	field := TextField{Name: "q", Options: []string{"yes", "no"}}
	row := TextRow{State: "state", Menus: [][]string{nil}}
	encode := func(s string) ([]int, error) { return []int{len(s)}, nil }
	for name, tc := range map[string]struct {
		rows                           []TextRow
		fields                         []TextField
		stateLimit, questionLimit, pad int
		enc                            TextEncoder
	}{
		"nil encoder":         {[]TextRow{row}, []TextField{field}, 8, 8, 0, nil},
		"no rows":             {nil, []TextField{field}, 8, 8, 0, encode},
		"no fields":           {[]TextRow{row}, nil, 8, 8, 0, encode},
		"empty state":         {[]TextRow{{Menus: row.Menus}}, []TextField{field}, 8, 8, 0, encode},
		"missing menu":        {[]TextRow{{State: row.State}}, []TextField{field}, 8, 8, 0, encode},
		"one candidate":       {[]TextRow{{State: row.State, Menus: [][]string{{"yes"}}}}, []TextField{field}, 8, 8, 0, encode},
		"duplicate candidate": {[]TextRow{{State: row.State, Menus: [][]string{{"yes", "yes"}}}}, []TextField{field}, 8, 8, 0, encode},
		"empty option":        {[]TextRow{row}, []TextField{{Name: "q", Options: []string{"yes", ""}}}, 8, 8, 0, encode},
		"duplicate option":    {[]TextRow{row}, []TextField{{Name: "q", Options: []string{"yes", "yes"}}}, 8, 8, 0, encode},
		"zero state limit":    {[]TextRow{row}, []TextField{field}, 0, 8, 0, encode},
		"zero prompt limit":   {[]TextRow{row}, []TextField{field}, 8, 0, 0, encode},
		"encoding fails":      {[]TextRow{row}, []TextField{field}, 8, 8, 0, func(string) ([]int, error) { return nil, errors.New("fail") }},
		"encoding empty":      {[]TextRow{row}, []TextField{field}, 8, 8, 0, func(string) ([]int, error) { return nil, nil }},
		"encoding oversized":  {[]TextRow{row}, []TextField{field}, 8, 8, 0, func(string) ([]int, error) { return make([]int, 4097), nil }},
		"negative ID":         {[]TextRow{row}, []TextField{field}, 8, 8, 0, func(string) ([]int, error) { return []int{-1}, nil }},
		"negative pad":        {[]TextRow{row}, []TextField{field}, 8, 8, -1, encode},
	} {
		t.Run(name, func(t *testing.T) {
			got, err := PackTextRows(tc.rows, tc.fields, tc.stateLimit, tc.questionLimit, tc.pad, tc.enc)
			if err == nil || got != nil {
				t.Fatalf("accepted malformed text: %v", got)
			}
		})
	}
	prompt, err := renderChoicePrompt(TextField{Name: "many_options", Options: []string{"a", "b", "c", "d", "e", "f", "g", "h", "i"}})
	if err != nil || strings.Contains(prompt, "options:") || !strings.HasPrefix(prompt, "many options | kind: choice") {
		t.Fatal("invalid >8-option prompt", prompt, err)
	}
}
