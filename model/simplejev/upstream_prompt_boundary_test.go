package simplejev

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
)

// This fixture contains only hashes and structural observations. The MIT Go
// package has no copy of the upstream Apache v1 prompt or chat-template text.
func TestPinnedUpstreamPromptBoundaryMetadata(t *testing.T) {
	for _, pin := range []struct{ path, sha string }{
		{"../../scripts/simplejev_oracle_prompt_boundary.py", "f4b00634ff186a9024e03ecabcd6998c948a3946310886bddecf7676f137b9cf"},
		{"testdata/upstream_prompt_boundary_v1.json", "f7011071f9690f70cba2eead7ad578009b8267f79ddf5b1f50d50c2cc025012b"},
	} {
		data, err := os.ReadFile(pin.path)
		if err != nil {
			t.Fatal(err)
		}
		if fmt.Sprintf("%x", sha256.Sum256(data)) != pin.sha {
			t.Fatalf("prompt audit hash changed: %s", pin.path)
		}
	}
	data, err := os.ReadFile("testdata/upstream_prompt_boundary_v1.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Schema    int               `json:"schema"`
		Revision  string            `json:"source_revision"`
		SourceSHA map[string]string `json:"source_sha256"`
		Tokenizer string            `json:"tokenizer"`
		Branches  []struct {
			ID          string   `json:"id"`
			TokenCount  int      `json:"token_count"`
			TokenSHA    string   `json:"token_sha256"`
			PrefixBytes int      `json:"answer_prefix_bytes"`
			PrefixSHA   string   `json:"answer_prefix_sha256"`
			Labels      []string `json:"labels"`
			TokenIDs    []int    `json:"label_token_ids"`
			Messages    int      `json:"messages"`
		} `json:"branches"`
		Failures []struct {
			Name     string `json:"name"`
			Category string `json:"category"`
		} `json:"failures"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	pins := map[string]string{
		"hf-server/hf_server.py":      "a96440b203d047800498479f85ecb417da340b77e837a5dc102fcb81938e3b1b",
		"hf-server/tests/conftest.py": "cafc09af4e292c1ed6fb797fdf725dadda5b692db9926fcc334ead4bb45d3611",
		"common/prompt_builder.py":    "14a885d3bfa44b0c8aa0ffc9ce84611a459957d938e1847f96d93c7f1840fa3d",
		"common/request_schema.py":    "6fa1c1215e8fc9de7aedbee77db6520bbb811922df9c45858bf78c4e4a02ceac",
	}
	if fixture.Schema != 1 || fixture.Revision != "b02aa81c915a8193759b3cd33fef74721d6e005b" || !reflect.DeepEqual(fixture.SourceSHA, pins) || fixture.Tokenizer != "upstream byte test double" || len(fixture.Branches) != 3 || len(fixture.Failures) != 2 {
		t.Fatal("unexpected prompt provenance")
	}
	for i, branch := range fixture.Branches {
		if branch.ID != fmt.Sprint(i) || branch.TokenCount <= 0 || len(branch.TokenSHA) != 64 || branch.PrefixBytes <= 0 || len(branch.PrefixSHA) != 64 || branch.Messages != 2 {
			t.Fatalf("branch %d geometry changed", i)
		}
		if len(branch.Labels) != len(branch.TokenIDs) {
			t.Fatalf("branch %d label count changed", i)
		}
		for n, label := range branch.Labels {
			if len(label) != 1 || branch.TokenIDs[n] != int(label[0]) {
				t.Fatalf("branch %d label boundary changed", i)
			}
		}
	}
	if len(fixture.Branches[0].Labels) != 2 || len(fixture.Branches[1].Labels) != 11 || len(fixture.Branches[2].Labels) != 9 || fixture.Branches[0].TokenCount != 1567 || fixture.Branches[1].TokenCount != 2469 || fixture.Branches[2].TokenCount != 1540 || fixture.Failures[0].Category != "single-token" || fixture.Failures[1].Category != "distinct" {
		t.Fatal("upstream prompt boundary observation changed")
	}
}
