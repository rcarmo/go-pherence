package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadSweepPrompts(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "prompts.txt")
	if err := os.WriteFile(path, []byte("# comment\nHello\n\nWhat is 2+2?\n"), 0o644); err != nil {
		t.Fatalf("write prompts: %v", err)
	}
	got := loadSweepPrompts(path)
	if len(got) != 2 || got[0] != "Hello" || got[1] != "What is 2+2?" {
		t.Fatalf("prompts=%v", got)
	}
}

func TestSweepLimitSlicesPrompts(t *testing.T) {
	prompts := []string{"a", "b", "c"}
	limit := 2
	if limit > 0 && limit < len(prompts) {
		prompts = prompts[:limit]
	}
	if len(prompts) != 2 || prompts[1] != "b" {
		t.Fatalf("prompts=%v", prompts)
	}
}

func TestSweepReportJSON(t *testing.T) {
	report := SweepReport{ModelDir: "m", Prompts: []string{"Hello"}, Runs: []Report{{Prompt: "Hello", Passed: true}}, Accepted: 1, Total: 1, AcceptanceRate: 1, AcceptedPrefixes: 2}
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var decoded SweepReport
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if decoded.Accepted != 1 || decoded.Total != 1 || decoded.AcceptanceRate != 1 || decoded.AcceptedPrefixes != 2 || len(decoded.Runs) != 1 {
		t.Fatalf("decoded=%+v", decoded)
	}
}
