package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSummarizeManifest(t *testing.T) {
	m := manifest{Version: 1, Families: map[string]manifestFamily{
		"b": {Status: "two", ValidationTarget: "make test", Coverage: map[string]bool{"done": true, "todo": false}},
		"a": {Status: "one", ValidationTarget: "make test", Coverage: map[string]bool{"done": true}},
	}}
	s, err := summarize(m, "", coverageFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(s) != 2 || s[0].Name != "a" || s[1].Name != "b" {
		t.Fatalf("summaries=%+v", s)
	}
	if s[1].Covered != 1 || s[1].Pending != 1 || len(s[1].PendingKeys) != 1 || s[1].PendingKeys[0] != "todo" {
		t.Fatalf("summary=%+v", s[1])
	}
	filtered, err := summarize(m, "b", coverageFilter{})
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered) != 1 || filtered[0].Name != "b" {
		t.Fatalf("filtered=%+v", filtered)
	}
	if _, err := summarize(m, "missing", coverageFilter{}); err == nil {
		t.Fatal("expected unknown family error")
	}
}

func TestSummarizeReferencesOnly(t *testing.T) {
	m := manifest{Version: 1, Families: map[string]manifestFamily{
		"x": {Status: "one", ValidationTarget: "make test", Coverage: map[string]bool{"config_parsing": true, "reference_coverage_reporting": true, "semantic_token_reference_fixture": false, "fixture_coverage_make_target": true}},
	}}
	s, err := summarize(m, "x", coverageFilter{ReferencesOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(s) != 1 || s[0].Covered != 2 || s[0].Pending != 1 || len(s[0].PendingKeys) != 1 || s[0].PendingKeys[0] != "semantic_token_reference_fixture" {
		t.Fatalf("summary=%+v", s)
	}
}

func TestSummarizeParityOnly(t *testing.T) {
	m := manifest{Version: 1, Families: map[string]manifestFamily{
		"x": {Status: "one", ValidationTarget: "make test", Coverage: map[string]bool{"config_parsing": true, "placeholder_reference_tracking": true, "numeric_parity_readiness_gate": true, "first_token_reference_fixture": false}},
	}}
	s, err := summarize(m, "x", coverageFilter{ParityOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(s) != 1 || s[0].Covered != 2 || s[0].Pending != 0 || len(s[0].PendingKeys) != 0 {
		t.Fatalf("summary=%+v", s)
	}
}

func TestSummarizeRuntimeOnly(t *testing.T) {
	m := manifest{Version: 1, Families: map[string]manifestFamily{
		"x": {Status: "one", ValidationTarget: "make test", Coverage: map[string]bool{"config_parsing": true, "runtime_status_reporting": true, "cpu_generation_runtime": false, "nvidia_runtime": false, "streaming_runtime": true}},
	}}
	s, err := summarize(m, "x", coverageFilter{RuntimeOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	wantPending := []string{"cpu_generation_runtime", "nvidia_runtime"}
	if len(s) != 1 || s[0].Covered != 2 || s[0].Pending != 2 || len(s[0].PendingKeys) != len(wantPending) {
		t.Fatalf("summary=%+v", s)
	}
	for i := range wantPending {
		if s[0].PendingKeys[i] != wantPending[i] {
			t.Fatalf("pending=%+v", s[0].PendingKeys)
		}
	}
}

func TestLoadManifest(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifest.json")
	m := manifest{Version: 1, Families: map[string]manifestFamily{"x": {Status: "ok", Coverage: map[string]bool{"a": true}}}}
	data, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatal(err)
	}
	loaded, err := loadManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Version != 1 || len(loaded.Families) != 1 {
		t.Fatalf("loaded=%+v", loaded)
	}
	if _, err := loadManifest(filepath.Join(dir, "missing.json")); err == nil {
		t.Fatal("expected missing manifest error")
	}
}
