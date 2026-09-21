package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/rcarmo/go-pherence/model/jevlike"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFinalMetricsKeepRejectionsInDenominator(t *testing.T) {
	rows := []directBatchInput{{GoldID: "a", Request: jevlike.DirectChoiceRequest{Candidates: []jevlike.ChoiceCandidate{{ID: "a", Text: "a"}, {ID: "b", Text: "b"}}}}, {GoldID: "b", Request: jevlike.DirectChoiceRequest{Candidates: []jevlike.ChoiceCandidate{{ID: "a", Text: "a"}, {ID: "b", Text: "b"}, {ID: "c", Text: "c"}}}}}
	outcomes := []directBatchOutput{{Result: &jevlike.DirectChoiceResult{IDs: []string{"a", "b"}, Logits: []float32{2, 0}, SelectedID: "a"}, Elapsed: 1}, {Error: "overlength", Elapsed: 0.01}}
	g, e := finalGroupMetrics("all", rows, outcomes, 2)
	if e != nil {
		t.Fatal(e)
	}
	if g.Requested != 2 || g.Admitted != 1 || g.Rejected != 1 || g.AccuracyCountingRejectionsWrong != .5 || g.Raw.Accuracy != 1 || g.Calibrated.Accuracy != 1 || g.RejectionCauses["overlength"] != 1 {
		t.Fatal(g)
	}
	if math.Abs(g.RandomAccuracyAllRequested-(.5+1.0/3)/2) > 1e-12 {
		t.Fatal(g.RandomAccuracyAllRequested)
	}
	if g.Raw.NLL == g.Calibrated.NLL {
		t.Fatal("calibration not applied")
	}
	if _, e = finalGroupMetrics("empty", nil, nil, 1); e == nil {
		t.Fatal("empty accepted")
	}
}

func TestFinalReportCompleteFrozenCohort(t *testing.T) {
	dir := t.TempDir()
	t.Chdir(dir)
	write := func(path string, value any) {
		t.Helper()
		if e := os.MkdirAll(filepath.Dir(path), 0o755); e != nil {
			t.Fatal(e)
		}
		b, e := json.Marshal(value)
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(path, append(b, '\n'), 0o600); e != nil {
			t.Fatal(e)
		}
	}
	digest := func(path string) string {
		t.Helper()
		b, e := os.ReadFile(path)
		if e != nil {
			t.Fatal(e)
		}
		return directHash(b)
	}
	a, b := directHash([]byte("yes"))[:16], directHash([]byte("no"))[:16]
	write("data/test.jsonl", map[string]any{"context": "Question", "options": []string{"yes", "no"}, "label": 0})
	write("data/provenance.jsonl", map[string]any{"partition": "test", "source_id": "row", "output_row": 1, "input": 0})
	write("data/manifest.json", map[string]any{"inputs": []any{map[string]string{"source": "task"}}, "artifacts": []any{map[string]string{"path": "test.jsonl", "sha256": digest("data/test.jsonl")}, map[string]string{"path": "provenance.jsonl", "sha256": digest("data/provenance.jsonl")}}})
	id := "synthetic#sha256=" + strings.Repeat("a", 64)
	contract := frozenFeatureContract(id, digest("data/manifest.json"), "f32")
	contract.Width = 4
	write("cache/contract.json", contract)
	write("assets.json", map[string]any{"synthetic": true})
	cal := func(path, model string) {
		write(path, map[string]any{"version": 2, "partition": "calibration", "temperature_fitted": true, "temperature": 2, "model_id": model, "source_manifest_sha256": digest("data/manifest.json")})
	}
	cal("direct-cal.json", id)
	f := finalFreeze{Version: 1, Scope: "final-once-normal-v1", Dataset: "data", DatasetSHA256: digest("data/manifest.json"), PartitionSHA256: digest("data/test.jsonl"), Expected: 1, ModelID: id, Assets: "assets.json", Contract: "cache/contract.json", DirectCalibration: "direct-cal.json", Files: map[string]string{}}
	models := map[string]string{"instruction": id}
	for _, seed := range []int{7, 17, 27} {
		ck := fmt.Sprintf("seed-%d.json", seed)
		cp := fmt.Sprintf("seed-%d-cal.json", seed)
		write(ck, map[string]any{"synthetic_checkpoint_identity": seed})
		mid := "jevlike-head@" + digest(ck) + "/feature:" + contract.ID()
		cal(cp, mid)
		f.Heads = append(f.Heads, struct {
			Seed        int    `json:"seed"`
			Checkpoint  string `json:"checkpoint"`
			Calibration string `json:"calibration"`
		}{seed, ck, cp})
		f.Files[ck] = digest(ck)
		f.Files[cp] = digest(cp)
		models[fmt.Sprintf("seed-%d", seed)] = mid
	}
	for _, p := range []string{"assets.json", "cache/contract.json", "direct-cal.json", "data/manifest.json"} {
		f.Files[p] = digest(p)
	}
	write("freeze.json", f)
	row := directBatchInput{ID: "row", Task: "task", Variant: "normal", GoldID: a, Request: jevlike.DirectChoiceRequest{Question: "Question", Candidates: []jevlike.ChoiceCandidate{{ID: a, Text: "yes"}, {ID: b, Text: "no"}}, Temperature: 1}}
	write("study/requests.jsonl", row)
	write("study/selection.json", map[string]any{"version": 2, "partition": "test", "requests": 1, "source_manifest_sha256": f.DatasetSHA256, "partition_sha256": f.PartitionSHA256, "request_sha256": digest("study/requests.jsonl"), "final_freeze_sha256": digest("freeze.json")})
	key := directRowKey(row.Task, row.ID, row.Variant)
	entry := finalEntry{Version: 1, FreezeSHA256: digest("freeze.json"), RequestSHA256: digest("study/requests.jsonl"), RequestKey: key, Outcomes: map[string]directBatchOutput{}}
	for arm, id := range models {
		entry.Outcomes[arm] = directBatchOutput{ID: row.ID, Task: row.Task, Variant: row.Variant, GoldID: row.GoldID, ModelID: id, Result: &jevlike.DirectChoiceResult{IDs: []string{a, b}, Logits: []float32{1, 0}, SelectedID: a}}
	}
	os.Mkdir("records", 0o755)
	args := []string{"-freeze", "freeze.json", "-study", "study", "-records", "records", "-output", "report"}
	var out, stderr bytes.Buffer
	if e := runFinalReport(args, &out, &stderr); e == nil {
		t.Fatal("incomplete cohort accepted")
	}
	path := filepath.Join("records", "0001-"+directHash([]byte(key))[:16]+".json")
	if e := writeFinalEntry(path, entry); e != nil {
		t.Fatal(e)
	}
	os.Mkdir("records/.run.lock", 0o700)
	if e := runFinalReport(args, &out, &stderr); e == nil {
		t.Fatal("active run accepted")
	}
	os.Remove("records/.run.lock")
	if e := runFinalReport(args, &out, &stderr); e != nil {
		t.Fatal(e)
	}
	result, e := os.ReadFile("report/report.json")
	if e != nil {
		t.Fatal(e)
	}
	var summary struct {
		Requested int
		Common    int `json:"common_admitted"`
	}
	if e = json.Unmarshal(result, &summary); e != nil || summary.Requested != 1 || summary.Common != 1 {
		t.Fatal(summary, e)
	}
	if e = runFinalReport(args, &out, &stderr); e == nil {
		t.Fatal("report overwritten")
	}
	write("direct-cal.json", map[string]any{"temperature": 3})
	args[len(args)-1] = "changed-report"
	if e = runFinalReport(args, &out, &stderr); e == nil {
		t.Fatal("changed calibration admitted")
	}
}
