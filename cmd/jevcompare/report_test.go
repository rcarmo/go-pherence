package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func writeJSONLines(t *testing.T, path string, rows ...any) []byte {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	encoder := json.NewEncoder(file)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			file.Close()
			t.Fatal(err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	bytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return bytes
}

func TestRunScoreReport(t *testing.T) {
	dir := t.TempDir()
	study := filepath.Join(dir, "study")
	if err := os.Mkdir(study, 0o755); err != nil {
		t.Fatal(err)
	}
	first := input{ID: "one", Task: "task/a", Variant: "normal", GoldID: "a", Request: request{Question: "q1", Temperature: 1, Candidates: []candidate{{ID: "a", Text: "yes"}, {ID: "b", Text: "no"}}}}
	second := input{ID: "two", Task: "task/b", Variant: "normal", GoldID: "b", Request: request{Question: "q2", Temperature: 1, Candidates: []candidate{{ID: "a", Text: "yes"}, {ID: "b", Text: "no"}}}}
	requests := writeJSONLines(t, filepath.Join(study, "screening.jsonl"), first, second)
	selectionBytes, err := json.Marshal(selection{Scope: "jev-port-bakeoff-v1", Cohorts: map[string]struct {
		Path   string `json:"path"`
		Rows   int    `json:"rows"`
		SHA256 string `json:"sha256"`
	}{"screening": {Path: "screening.jsonl", Rows: 2, SHA256: digest(requests)}}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(study, "selection.json"), selectionBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	resultsPath := filepath.Join(dir, "results.jsonl")
	writeJSONLines(t, resultsPath,
		output{Version: 1, Arm: "test", ID: first.ID, Task: first.Task, Variant: first.Variant, GoldID: first.GoldID, ModelID: "model", SelectedID: "a", DecisionSeconds: 1, Options: []optionScore{{ID: "a", Text: "yes", Score: 2}, {ID: "b", Text: "no", Score: 1}}},
		output{Version: 1, Arm: "test", ID: second.ID, Task: second.Task, Variant: second.Variant, GoldID: second.GoldID, ModelID: "model", SelectedID: "a", DecisionSeconds: 2, Options: []optionScore{{ID: "a", Text: "yes", Score: 0.9}, {ID: "b", Text: "no", Score: 0.1}}},
	)
	reportPath := filepath.Join(dir, "report.json")
	if err := runScoreReport(study, "screening", resultsPath, reportPath); err != nil {
		t.Fatal(err)
	}
	bytes, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatal(err)
	}
	var report scoreReport
	if err := json.Unmarshal(bytes, &report); err != nil {
		t.Fatal(err)
	}
	if report.TemperatureFitted || report.Arm != "test" || report.ModelID != "model" || len(report.Groups) != 3 {
		t.Fatalf("unexpected report %+v", report)
	}
	var all *scoreReportGroup
	for i := range report.Groups {
		if report.Groups[i].Group == "all" && report.Groups[i].Variant == "normal" {
			all = &report.Groups[i]
		}
	}
	if all == nil || all.Admitted != 2 || all.Rejected != 0 || all.AccuracyCountingRejectionsWrong != 0.5 || all.MetricsAdmitted == nil {
		t.Fatalf("unexpected all group %+v", all)
	}
	wantNLL := (-math.Log(2.0/3) - math.Log(0.1)) / 2
	if math.Abs(all.MetricsAdmitted.NLL-wantNLL) > 1e-6 || all.MetricsAdmitted.Accuracy != 0.5 {
		t.Fatalf("unexpected metrics %+v", all.MetricsAdmitted)
	}
}

func TestRunScoreReportRejectsChangedResultIdentity(t *testing.T) {
	dir := t.TempDir()
	study := filepath.Join(dir, "study")
	if err := os.Mkdir(study, 0o755); err != nil {
		t.Fatal(err)
	}
	in := input{ID: "one", Task: "task", Variant: "normal", GoldID: "a", Request: request{Question: "q", Temperature: 1, Candidates: []candidate{{ID: "a", Text: "yes"}, {ID: "b", Text: "no"}}}}
	requests := writeJSONLines(t, filepath.Join(study, "screening.jsonl"), in)
	selectionBytes, _ := json.Marshal(selection{Scope: "jev-port-bakeoff-v1", Cohorts: map[string]struct {
		Path   string `json:"path"`
		Rows   int    `json:"rows"`
		SHA256 string `json:"sha256"`
	}{"screening": {Path: "screening.jsonl", Rows: 1, SHA256: digest(requests)}}})
	if err := os.WriteFile(filepath.Join(study, "selection.json"), selectionBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	results := filepath.Join(dir, "results.jsonl")
	writeJSONLines(t, results, output{Version: 1, Arm: "test", ID: "wrong", Task: in.Task, Variant: in.Variant, GoldID: in.GoldID, ModelID: "model", SelectedID: "a", Options: []optionScore{{ID: "a", Text: "yes", Score: 1}, {ID: "b", Text: "no", Score: 0}}})
	if err := runScoreReport(study, "screening", results, filepath.Join(dir, "report.json")); err == nil {
		t.Fatal("changed result identity accepted")
	}
}

func TestValidateOutputRejectsSelectedScoreMismatch(t *testing.T) {
	in := input{ID: "one", Task: "task", Variant: "normal", GoldID: "a", Request: request{Question: "q", Temperature: 1, Candidates: []candidate{{ID: "a", Text: "yes"}, {ID: "b", Text: "no"}}}}
	out := output{Version: 1, Arm: "test", ID: in.ID, Task: in.Task, Variant: in.Variant, GoldID: in.GoldID, ModelID: "model", SelectedID: "a", Options: []optionScore{{ID: "a", Text: "yes", Score: 0.1}, {ID: "b", Text: "no", Score: 0.9}}}
	if err := validateOutput(out, in, "test", "model"); err == nil {
		t.Fatal("selected ID inconsistent with scores was accepted")
	}
}
