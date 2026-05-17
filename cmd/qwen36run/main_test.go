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

func TestApplySweepLimit(t *testing.T) {
	prompts := []string{"a", "b", "c"}
	cases := []struct {
		limit int
		want  int
	}{
		{0, 3},
		{-1, 3},
		{2, 2},
		{3, 3},
		{4, 3},
	}
	for _, tc := range cases {
		got := applySweepLimit(prompts, tc.limit)
		if len(got) != tc.want {
			t.Fatalf("limit=%d len=%d want %d", tc.limit, len(got), tc.want)
		}
	}
	if got := applySweepLimit(prompts, 2); got[1] != "b" {
		t.Fatalf("limited prompts=%v", got)
	}
}

func TestSweepReportJSON(t *testing.T) {
	report := SweepReport{ModelDir: "m", Prompts: []string{"Hello"}, Runs: []Report{{Prompt: "Hello", DurationMS: 7, Passed: true}}, Accepted: 1, Total: 1, AcceptanceRate: 1, AcceptedPrefixes: 2, DurationMS: 9}
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var decoded SweepReport
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if decoded.Accepted != 1 || decoded.Total != 1 || decoded.AcceptanceRate != 1 || decoded.AcceptedPrefixes != 2 || decoded.DurationMS != 9 || len(decoded.Runs) != 1 || decoded.Runs[0].DurationMS != 7 {
		t.Fatalf("decoded=%+v", decoded)
	}
}
