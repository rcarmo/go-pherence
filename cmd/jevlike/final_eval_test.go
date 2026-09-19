package main

import (
	"encoding/json"
	"github.com/rcarmo/go-pherence/model/jevlike"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestFinalImmutableEntryAndResumeValidation(t *testing.T) {
	row := directBatchInput{ID: "a", Task: "task", Variant: "normal", GoldID: "yes", Request: jevlike.DirectChoiceRequest{Candidates: []jevlike.ChoiceCandidate{{ID: "yes", Text: "yes"}, {ID: "no", Text: "no"}}}}
	models := map[string]string{"instruction": "frozen"}
	entry := finalEntry{Version: 1, FreezeSHA256: "freeze", RequestSHA256: "request", RequestKey: directRowKey(row.Task, row.ID, row.Variant), Outcomes: map[string]directBatchOutput{"instruction": {ID: row.ID, Task: row.Task, Variant: row.Variant, GoldID: row.GoldID, ModelID: "frozen", Result: &jevlike.DirectChoiceResult{IDs: []string{"yes", "no"}, Logits: []float32{1, 0}, SelectedID: "yes"}}}}
	if e := validateFinalEntry(entry, "freeze", "request", row, models); e != nil {
		t.Fatal(e)
	}
	path := filepath.Join(t.TempDir(), "out.json")
	if e := writeFinalEntry(path, entry); e != nil {
		t.Fatal(e)
	}
	if e := writeFinalEntry(path, entry); e == nil {
		t.Fatal("overwrote completed entry")
	}
	b, e := os.ReadFile(path)
	if e != nil {
		t.Fatal(e)
	}
	var decoded finalEntry
	if e = json.Unmarshal(b, &decoded); e != nil {
		t.Fatal(e)
	}
	if e = validateFinalEntry(decoded, "freeze", "request", row, models); e != nil {
		t.Fatal(e)
	}
	if e = validateFinalEntry(decoded, "changed", "request", row, models); e == nil {
		t.Fatal("wrong freeze")
	}
	if e = validateFinalEntry(decoded, "freeze", "changed", row, models); e == nil {
		t.Fatal("wrong request set")
	}
	x := decoded.Outcomes["instruction"]
	x.Result.Logits[0] = float32(math.NaN())
	if e = validateFinalEntry(decoded, "freeze", "request", row, models); e == nil {
		t.Fatal("nonfinite accepted")
	}
	x.Result = nil
	x.Error = "overlength"
	decoded.Outcomes["instruction"] = x
	if e = validateFinalEntry(decoded, "freeze", "request", row, models); e != nil {
		t.Fatal("rejection not accepted", e)
	}
	delete(decoded.Outcomes, "instruction")
	if e = validateFinalEntry(decoded, "freeze", "request", row, models); e == nil {
		t.Fatal("incomplete outcomes")
	}
}
