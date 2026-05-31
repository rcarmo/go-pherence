package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
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
	if s[1].Categories == nil {
		t.Fatalf("missing categories: %+v", s[1])
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

func TestSummarizeCategoryCounts(t *testing.T) {
	m := manifest{Version: 1, Families: map[string]manifestFamily{
		"x": {Status: "one", ValidationTarget: "make test", Coverage: map[string]bool{"reference_coverage_reporting": true, "semantic_token_reference_fixture": false, "runtime_status_reporting": true, "cpu_generation_runtime": false, "numeric_parity_readiness_gate": true, "execution_readiness_gate": true}},
	}}
	s, err := summarize(m, "x", coverageFilter{})
	if err != nil {
		t.Fatal(err)
	}
	cats := s[0].Categories
	if cats["references"].Covered != 1 || cats["references"].Pending != 1 || cats["references"].PendingKeys[0] != "semantic_token_reference_fixture" {
		t.Fatalf("reference category=%+v", cats["references"])
	}
	if cats["runtime"].Covered != 1 || cats["runtime"].Pending != 1 || cats["runtime"].PendingKeys[0] != "cpu_generation_runtime" {
		t.Fatalf("runtime category=%+v", cats["runtime"])
	}
	if cats["parity"].Covered != 1 || cats["parity"].Pending != 0 {
		t.Fatalf("parity category=%+v", cats["parity"])
	}
	if cats["readiness"].Covered != 2 || cats["readiness"].Pending != 0 {
		t.Fatalf("readiness category=%+v", cats["readiness"])
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

func TestSummarizeReadinessOnly(t *testing.T) {
	m := manifest{Version: 1, Families: map[string]manifestFamily{
		"x": {Status: "one", ValidationTarget: "make test", Coverage: map[string]bool{"config_parsing": true, "runtime_readiness_report": true, "execution_readiness_gate": true, "ready_for_execution_fixture": false}},
	}}
	s, err := summarize(m, "x", coverageFilter{ReadinessOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(s) != 1 || s[0].Covered != 2 || s[0].Pending != 1 || len(s[0].PendingKeys) != 1 || s[0].PendingKeys[0] != "ready_for_execution_fixture" {
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

func TestPrintTextSummaryIncludesCategories(t *testing.T) {
	var buf bytes.Buffer
	s := familySummary{Name: "x", Status: "ok", Covered: 2, Pending: 1, Categories: map[string]categoryCounts{
		"references": {Covered: 1},
		"runtime":    {Pending: 1, PendingKeys: []string{"cpu_runtime"}},
		"parity":     {Covered: 1},
		"readiness":  {},
	}}
	printTextSummary(&buf, s)
	out := buf.String()
	for _, want := range []string{"x: ok covered=2 pending=1", "runtime: covered=0 pending=1 pending_keys=[cpu_runtime]", "readiness: covered=0 pending=0"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
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
