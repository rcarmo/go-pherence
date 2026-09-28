package qwen3tts

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestRustCPUReleasedPrefillSamplingFixture(t *testing.T) {
	root := filepath.Join("testdata", "customvoice_0b6_ryan_hello")
	var ref struct {
		OracleRevision string `json:"oracle_revision"`
		LogitsSHA      string `json:"logits_sha256"`
		ScriptSHA      string `json:"released_first_sample_oracle_script_sha256"`
		SampleSHA      string `json:"released_first_sample_sha256"`
	}
	data, err := os.ReadFile(filepath.Join(root, "reference.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &ref); err != nil {
		t.Fatal(err)
	}
	if ref.OracleRevision != "711ceee07cad92673f86de8997bdf54c30caa49f" || ref.LogitsSHA != "a21ec2ea111681b69195b8bedbdd089a5ac18a64eb7eec32b76fa3cf34c90149" || ref.ScriptSHA != "a55f59266e2a4cb3c6abb230c0b26f3bcff461c8833bd6b1c29f6633b00651f9" || ref.SampleSHA != "16db08594d3f353956959a127f2eacd4b226c095268b0b3fa6d4a7ea6bce7fba" {
		t.Fatal("unexpected released sampler provenance")
	}
	for _, file := range []struct{ path, hash string }{
		{filepath.Join(root, "logits.f32le"), ref.LogitsSHA},
		{filepath.Join(root, "released_first_sample.json"), ref.SampleSHA},
		{filepath.Join("..", "..", "scripts", "qwen3tts_oracle_released_first_sample.rs"), ref.ScriptSHA},
	} {
		if err := verifyReleasedFile(file.path, file.hash, 0); err != nil {
			t.Fatal(err)
		}
	}
	data, err = os.ReadFile(filepath.Join(root, "released_first_sample.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		LogitsSHA  string `json:"logits_sha256"`
		Controlled struct {
			TokenID     uint32 `json:"token_id"`
			Replacement string `json:"replacement"`
		} `json:"controlled_change"`
		Rows []struct {
			Label       string   `json:"label"`
			Seed        uint64   `json:"seed"`
			Temperature float64  `json:"temperature"`
			TopK        int      `json:"top_k"`
			TopP        float64  `json:"top_p"`
			Tokens      []uint32 `json:"tokens"`
		} `json:"rows"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.LogitsSHA != ref.LogitsSHA || fixture.Controlled.TokenID != 1221 || fixture.Controlled.Replacement != "f32(logit[1995] - 0.25)" || len(fixture.Rows) != 12 {
		t.Fatal("invalid sampler fixture geometry")
	}
	near := map[uint64][]uint32{42: {1221, 1221, 1221, 1995, 1995, 1221, 1221, 1995}, 7: {1995, 1995, 1995, 1995, 1221, 1995, 1995, 1221}}
	seen := make(map[string]bool)
	for _, row := range fixture.Rows {
		key := fmt.Sprintf("%s/%d/%g/%d/%g", row.Label, row.Seed, row.Temperature, row.TopK, row.TopP)
		if seen[key] {
			t.Fatalf("duplicate sampler row %s", key)
		}
		seen[key] = true
		if row.Seed != 42 && row.Seed != 7 {
			t.Fatalf("seed=%d", row.Seed)
		}
		if !((row.Temperature == 0.7 && row.TopK == 50 && row.TopP == 0.9) || (row.Temperature == 0 && row.TopK == 50 && row.TopP == 0.9) || (row.Temperature == 0.7 && row.TopK == 0 && row.TopP == 1)) {
			t.Fatalf("unexpected config %s", key)
		}
		want := []uint32{1995, 1995, 1995, 1995, 1995, 1995, 1995, 1995}
		if row.Label == "controlled_near_tie" && row.Temperature == 0.7 {
			want = near[row.Seed]
		} else if row.Label != "controlled_near_tie" && row.Label != "released" {
			t.Fatalf("label=%q", row.Label)
		}
		if !reflect.DeepEqual(row.Tokens, want) {
			t.Fatalf("%s seed=%d tokens=%v want=%v", row.Label, row.Seed, row.Tokens, want)
		}
	}
	if len(seen) != 12 {
		t.Fatalf("distinct sampler rows=%d", len(seen))
	}
}
