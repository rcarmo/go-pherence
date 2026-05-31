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
	if s[0].CoveragePercent != 4.0/6.0*100.0 {
		t.Fatalf("coverage percent=%g", s[0].CoveragePercent)
	}
	if cats["references"].Covered != 1 || cats["references"].Pending != 1 || cats["references"].Percent != 50 || cats["references"].PendingKeys[0] != "semantic_token_reference_fixture" {
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

func TestSummarizeExecutionOnly(t *testing.T) {
	m := manifest{Version: 1, Families: map[string]manifestFamily{
		"x": {Status: "one", ValidationTarget: "make test", Coverage: map[string]bool{"runtime_status_reporting": true, "cpu_generation_runtime": false, "nvidia_runtime": false, "streaming_runtime": true}},
	}}
	s, err := summarize(m, "x", coverageFilter{ExecutionOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	wantPending := []string{"cpu_generation_runtime", "nvidia_runtime"}
	if len(s) != 1 || s[0].Covered != 1 || s[0].Pending != 2 || len(s[0].PendingKeys) != len(wantPending) {
		t.Fatalf("summary=%+v", s)
	}
	for i := range wantPending {
		if s[0].PendingKeys[i] != wantPending[i] {
			t.Fatalf("pending=%+v", s[0].PendingKeys)
		}
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

func TestPrintSnapshot(t *testing.T) {
	var buf bytes.Buffer
	summaries := []familySummary{{Name: "x", Status: "ok", Covered: 1, Pending: 1, CoveragePercent: 50, Categories: map[string]categoryCounts{
		"references": {Covered: 1, Percent: 100},
		"runtime":    {Pending: 1, PendingKeys: []string{"cpu_talker_runtime"}},
		"execution":  {Pending: 1, PendingKeys: []string{"cpu_talker_runtime"}},
		"parity":     {Percent: 100},
		"readiness":  {Percent: 100},
	}}}
	printSnapshot(&buf, summaries)
	out := buf.String()
	for _, want := range []string{"# Model coverage snapshot", "| x | ok | 1 | 1 | 50.0%", "# Runtime roadmap", "cpu_talker_runtime"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}

func TestBuildNextRuntime(t *testing.T) {
	summaries := []familySummary{{Name: "x", Categories: map[string]categoryCounts{"runtime": {PendingKeys: []string{"nvidia_runtime", "cpu_talker_runtime"}}}}}
	next := buildNextRuntime(summaries)
	if len(next) != 1 || next[0].Family != "x" || len(next[0].Blockers) != 1 || next[0].Blockers[0].Key != "cpu_talker_runtime" {
		t.Fatalf("next=%+v", next)
	}
}

func TestPrintNextRuntime(t *testing.T) {
	var buf bytes.Buffer
	summaries := []familySummary{{Name: "x", Categories: map[string]categoryCounts{"runtime": {PendingKeys: []string{"nvidia_runtime", "cpu_talker_runtime"}}}}}
	printNextRuntime(&buf, summaries)
	out := buf.String()
	for _, want := range []string{"x.cpu_talker_runtime", "implement the Qwen3-TTS CPU/reference Talker", "package: model/qwen3tts", "validate: cmd/qwen3ttsinspect -require-numeric-parity"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}

func TestBuildRuntimeRoadmap(t *testing.T) {
	summaries := []familySummary{{Name: "x", Categories: map[string]categoryCounts{"runtime": {PendingKeys: []string{"nvidia_runtime", "cpu_talker_runtime"}}}}}
	roadmap := buildRuntimeRoadmap(summaries)
	if len(roadmap) != 1 || roadmap[0].Family != "x" || len(roadmap[0].Blockers) != 2 {
		t.Fatalf("roadmap=%+v", roadmap)
	}
	if roadmap[0].Blockers[0].Key != "cpu_talker_runtime" || roadmap[0].Blockers[0].Phase != 10 || roadmap[0].Blockers[0].Package != "model/qwen3tts" || roadmap[0].Blockers[0].Validation == "" {
		t.Fatalf("blockers=%+v", roadmap[0].Blockers)
	}
	if roadmap[0].Blockers[1].Key != "nvidia_runtime" || roadmap[0].Blockers[1].Prerequisites == "" {
		t.Fatalf("blockers=%+v", roadmap[0].Blockers)
	}
}

func TestPrintRuntimeRoadmap(t *testing.T) {
	var buf bytes.Buffer
	summaries := []familySummary{{Name: "x", Categories: map[string]categoryCounts{"runtime": {PendingKeys: []string{"nvidia_runtime", "cpu_talker_runtime", "decoder12hz_runtime"}}}}}
	printRuntimeRoadmap(&buf, summaries)
	out := buf.String()
	for _, want := range []string{"## x runtime blockers", "- [ ] P10 `cpu_talker_runtime` — implement the Qwen3-TTS CPU/reference Talker semantic-token path", "_(package: `model/qwen3tts`)_", "_(validate: `cmd/qwen3ttsinspect -require-numeric-parity`)_", "- [ ] P30 `decoder12hz_runtime`", "_(after: cpu_code_predictor_runtime)_", "- [ ] P90 `nvidia_runtime` — add NVIDIA acceleration after CPU/reference parity is established _(package: `backends/nvidia`)_ _(after: CPU/reference parity)_"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
	if strings.Index(out, "cpu_talker_runtime") > strings.Index(out, "decoder12hz_runtime") || strings.Index(out, "decoder12hz_runtime") > strings.Index(out, "nvidia_runtime") {
		t.Fatalf("runtime blockers not in dependency order:\n%s", out)
	}
}

func TestSummariesMeetMinPercent(t *testing.T) {
	summaries := []familySummary{{Name: "a", CoveragePercent: 95}, {Name: "b", CoveragePercent: 90.6}}
	if !summariesMeetMinPercent(summaries, 90) {
		t.Fatal("expected threshold to pass")
	}
	if summariesMeetMinPercent(summaries, 91) {
		t.Fatal("expected threshold to fail")
	}
}

func TestPrintCSVSummary(t *testing.T) {
	var buf bytes.Buffer
	s := familySummary{Name: "x", Status: "ok", Covered: 3, Pending: 1, CoveragePercent: 75, Categories: map[string]categoryCounts{
		"references": {Covered: 1, Percent: 100},
		"runtime":    {Covered: 1, Pending: 1, Percent: 50},
		"execution":  {Pending: 1, Percent: 0},
		"parity":     {Covered: 1, Percent: 100},
		"readiness":  {Covered: 2, Percent: 100},
	}}
	printCSVSummary(&buf, []familySummary{s})
	out := buf.String()
	for _, want := range []string{"family,status,covered,pending,coverage_percent,references_covered", "x,ok,3,1,75.0,1,0,100.0,1,1,50.0,0,1,0.0,1,0,100.0,2,0,100.0"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}

func TestPrintMarkdownSummary(t *testing.T) {
	var buf bytes.Buffer
	s := familySummary{Name: "x", Status: "ok", Covered: 3, Pending: 1, CoveragePercent: 75, Categories: map[string]categoryCounts{
		"references": {Covered: 1, Percent: 100},
		"runtime":    {Covered: 1, Pending: 1, Percent: 50},
		"execution":  {Pending: 1, Percent: 0},
		"parity":     {Covered: 1, Percent: 100},
		"readiness":  {Covered: 2, Percent: 100},
	}}
	printMarkdownSummary(&buf, []familySummary{s})
	out := buf.String()
	for _, want := range []string{"| family | status | covered | pending | coverage | references | runtime | execution | parity | readiness |", "| x | ok | 3 | 1 | 75.0% | 1/0 (100.0%) | 1/1 (50.0%) | 0/1 (0.0%) | 1/0 (100.0%) | 2/0 (100.0%) |"} {
		if !strings.Contains(out, want) {
			t.Fatalf("missing %q in:\n%s", want, out)
		}
	}
}

func TestPrintTextSummaryIncludesCategories(t *testing.T) {
	var buf bytes.Buffer
	s := familySummary{Name: "x", Status: "ok", Covered: 2, Pending: 1, CoveragePercent: 66.7, Categories: map[string]categoryCounts{
		"references": {Covered: 1, Percent: 100},
		"runtime":    {Pending: 1, Percent: 0, PendingKeys: []string{"cpu_runtime"}},
		"execution":  {Pending: 1, Percent: 0, PendingKeys: []string{"cpu_runtime"}},
		"parity":     {Covered: 1, Percent: 100},
		"readiness":  {Percent: 100},
	}}
	printTextSummary(&buf, s)
	out := buf.String()
	for _, want := range []string{"x: ok covered=2 pending=1 coverage=66.7%", "runtime: covered=0 pending=1 coverage=0.0% pending_keys=[cpu_runtime]", "readiness: covered=0 pending=0 coverage=100.0%"} {
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
