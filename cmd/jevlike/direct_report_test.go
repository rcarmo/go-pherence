package main

import (
	"bytes"
	"encoding/json"
	"github.com/rcarmo/go-pherence/model/jevlike"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func reportFixture(t *testing.T) ([]string, directBatchOutput) {
	t.Helper()
	dir := t.TempDir()
	write := func(name string, b []byte) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	partition := []byte(`{"options":["yes","no"],"label":0}` + "\n")
	prov := []byte(`{"partition":"validation","source_id":"row","input":0,"output_row":1}` + "\n")
	manifest, _ := json.Marshal(map[string]any{"inputs": []any{map[string]string{"source": "task"}}, "artifacts": []any{map[string]string{"path": "validation.jsonl", "sha256": directHash(partition)}, map[string]string{"path": "calibration.jsonl", "sha256": directHash(partition)}, map[string]string{"path": "provenance.jsonl", "sha256": directHash(prov)}}})
	a, b := directHash([]byte("yes"))[:16], directHash([]byte("no"))[:16]
	req := directBatchInput{ID: "row", Task: "task", Variant: "normal", GoldID: a, Request: jevlike.DirectChoiceRequest{Question: "question", Candidates: []jevlike.ChoiceCandidate{{ID: a, Text: "yes"}, {ID: b, Text: "no"}}, Temperature: 1}}
	requests, _ := json.Marshal(req)
	requests = append(requests, '\n')
	selection, _ := json.Marshal(directSelection{Partition: "validation", Requests: 1, RequestSHA256: directHash(requests), SourceManifestSHA256: directHash(manifest), PartitionSHA256: directHash(partition)})
	row := directBatchOutput{ID: "row", Task: "task", Variant: "normal", GoldID: a, ModelID: "pinned", Result: &jevlike.DirectChoiceResult{IDs: []string{a, b}, Logits: []float32{1, 0}, SelectedID: a}}
	results, _ := json.Marshal(row)
	write("validation.jsonl", partition)
	write("calibration.jsonl", partition)
	write("provenance.jsonl", prov)
	write("manifest.json", manifest)
	write("requests.jsonl", requests)
	write("selection.json", selection)
	write("results.jsonl", append(results, '\n'))
	return []string{"-results", filepath.Join(dir, "results.jsonl"), "-selection", filepath.Join(dir, "selection.json"), "-requests", filepath.Join(dir, "requests.jsonl"), "-dataset", dir}, row
}
func TestDirectReportPartitionAndCompletion(t *testing.T) {
	args, _ := reportFixture(t)
	var out, stderr bytes.Buffer
	if err := runDirectReport(args, &out, &stderr); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"accuracy": 1`) {
		t.Fatal(out.String())
	}
	if err := runDirectReport(append(args, "-fit-temperature"), &out, &stderr); err == nil {
		t.Fatal("fit validation leaked")
	}
	b, err := os.ReadFile(args[3])
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(args[3], bytes.Replace(b, []byte(`"requests":1`), []byte(`"requests":2`), 1), 0o600); err != nil {
		t.Fatal(err)
	}
	if err = runDirectReport(args, &out, &stderr); err == nil {
		t.Fatal("partial accepted")
	}
}
func TestDirectReportRejectsRelabelledPartitionAndChangedResults(t *testing.T) {
	for _, test := range []string{"relabel-partition", "request-hash", "missing-result", "contradictory-result", "wrong-argmax", "wrong-id", "wrong-gold", "empty-model", "duplicate", "incomplete"} {
		t.Run(test, func(t *testing.T) {
			args, row := reportFixture(t)
			var out, stderr bytes.Buffer
			switch test {
			case "relabel-partition":
				b, _ := os.ReadFile(args[3])
				os.WriteFile(args[3], bytes.ReplaceAll(b, []byte("validation"), []byte("calibration")), 0o600)
				args = append(args, "-fit-temperature")
			case "request-hash":
				os.WriteFile(args[5], []byte("{}\n"), 0o600)
			default:
				switch test {
				case "missing-result":
					row.Result = nil
				case "contradictory-result":
					row.Error = "bad"
				case "wrong-argmax":
					row.Result.SelectedID = row.Result.IDs[1]
				case "wrong-id":
					row.ID = "foreign"
				case "wrong-gold":
					row.GoldID = row.Result.IDs[1]
				case "empty-model":
					row.ModelID = ""
				}
				b, _ := json.Marshal(row)
				b = append(b, '\n')
				if test == "duplicate" {
					b = append(b, b...)
				}
				if test == "incomplete" {
					b = nil
				}
				os.WriteFile(args[1], b, 0o600)
			}
			if err := runDirectReport(args, &out, &stderr); err == nil {
				t.Fatal("invalid report accepted")
			}
		})
	}
}

func TestDirectFrozenCalibrationIdentity(t *testing.T) {
	args, _ := reportFixture(t)
	selectionBytes, _ := os.ReadFile(args[3])
	var selection directSelection
	json.Unmarshal(selectionBytes, &selection)
	path := filepath.Join(args[7], "calibration-report.json")
	cal := map[string]any{"version": 2, "partition": "calibration", "temperature_fitted": true, "temperature": 2, "model_id": "pinned", "source_manifest_sha256": selection.SourceManifestSHA256, "request_sha256": strings.Repeat("b", 64), "result_sha256": strings.Repeat("c", 64)}
	write := func() {
		b, _ := json.Marshal(cal)
		if err := os.WriteFile(path, b, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write()
	var out, stderr bytes.Buffer
	full := append(args, "-calibration", path)
	if err := runDirectReport(full, &out, &stderr); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"temperature": 2`) {
		t.Fatal(out.String())
	}
	cal["model_id"] = "wrong same-width checkpoint"
	write()
	if err := runDirectReport(full, &out, &stderr); err == nil {
		t.Fatal("different checkpoint calibration accepted")
	}
	cal["model_id"] = "pinned"
	cal["partition"] = "validation"
	write()
	if err := runDirectReport(full, &out, &stderr); err == nil {
		t.Fatal("validation calibration accepted")
	}
}
