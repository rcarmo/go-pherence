package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestDirectStudyCLIAdmissionWithoutAssets(t *testing.T) {
	for _, command := range []string{"score-batch", "direct-report", "direct-bench"} {
		t.Run(command, func(t *testing.T) {
			var out, stderr bytes.Buffer
			if err := run([]string{command, "-h"}, &out, &stderr); err != nil {
				t.Fatal(err)
			}
			if err := run([]string{command}, &out, &stderr); err == nil {
				t.Fatal("missing arguments admitted")
			}
		})
	}
	// Failure before opening the destination cannot replace prior results.
	dir := t.TempDir()
	path := filepath.Join(dir, "existing-results.jsonl")
	if err := os.WriteFile(path, []byte("retained"), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	err := runScoreBatch([]string{"-encoder-model", dir, "-verified-assets", filepath.Join(dir, "missing.json"), "-requests", "missing.jsonl", "-output", path}, &out, &stderr)
	if err == nil {
		t.Fatal("missing identity admitted")
	}
	b, err := os.ReadFile(path)
	if err != nil || string(b) != "retained" {
		t.Fatal("existing results changed", err)
	}
}
