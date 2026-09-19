package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPreparePinnedSourcesAndArtifacts(t *testing.T) {
	dir := t.TempDir()
	var rows bytes.Buffer
	for i := 0; i < 100; i++ {
		fmt.Fprintf(&rows, "{\"id\":\"%d\",\"question\":\"question %d\",\"choices\":{\"label\":[\"A\",\"B\"],\"text\":[\"yes\",\"no\"]},\"answerKey\":\"A\"}\n", i, i)
	}
	path := filepath.Join(dir, "rows.jsonl")
	if err := os.WriteFile(path, rows.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(rows.Bytes())
	spec := datasetPrepareSpec{1, []datasetInput{{Source: "commonsense_qa", Config: "default", Revision: strings.Repeat("a", 40), License: "MIT", Split: "train", Path: "rows.jsonl", SHA256: hex.EncodeToString(h[:])}}}
	manifest := filepath.Join(dir, "inputs.json")
	b, _ := json.Marshal(spec)
	os.WriteFile(manifest, b, 0o600)
	out := filepath.Join(dir, "prepared")
	var stdout, stderr bytes.Buffer
	if err := run([]string{"prepare", "-manifest", manifest, "-output-dir", out}, &stdout, &stderr); err != nil {
		t.Fatal(err, stderr.String())
	}
	for _, name := range []string{"train.jsonl", "validation.jsonl", "calibration.jsonl", "test.jsonl", "provenance.jsonl", "manifest.json"} {
		if _, err := os.Stat(filepath.Join(out, name)); err != nil {
			t.Fatal(err)
		}
	}
	if !strings.Contains(stdout.String(), "spec_sha256") {
		t.Fatal(stdout.String())
	}
	if err := runPrepare([]string{"-manifest", manifest, "-output-dir", out}, &stdout, &stderr); err == nil {
		t.Fatal("overwrote dataset")
	}
	if err := runPrepare([]string{"-manifest", manifest, "-output-dir", out + "-repeat"}, &stdout, &stderr); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"train.jsonl", "validation.jsonl", "calibration.jsonl", "test.jsonl", "provenance.jsonl", "manifest.json"} {
		a, _ := os.ReadFile(filepath.Join(out, name))
		b, _ := os.ReadFile(filepath.Join(out+"-repeat", name))
		if !bytes.Equal(a, b) {
			t.Fatalf("nondeterministic %s", name)
		}
	}
	spec.Inputs[0].SHA256 = strings.Repeat("0", 64)
	b, _ = json.Marshal(spec)
	os.WriteFile(manifest, b, 0o600)
	if err := runPrepare([]string{"-manifest", manifest, "-output-dir", out + "-bad"}, &stdout, &stderr); err == nil || !strings.Contains(err.Error(), "SHA256") {
		t.Fatal(err)
	}
	if _, err := os.Stat(out + "-bad"); !os.IsNotExist(err) {
		t.Fatal("published partial output")
	}
}

func TestDatasetInputRequiresIdentityAndBoundary(t *testing.T) {
	base := datasetInput{Source: "multi_nli", Path: "input.jsonl", Revision: strings.Repeat("a", 40), SHA256: strings.Repeat("b", 64), License: "source terms", Split: "train"}
	for _, field := range []string{"revision", "hash", "licence", "split"} {
		in := base
		switch field {
		case "revision":
			in.Revision = "main"
		case "hash":
			in.SHA256 = ""
		case "licence":
			in.License = ""
		case "split":
			in.Split = ""
		}
		if err := validateDatasetInput(in); err == nil {
			t.Fatalf("accepted missing %s", field)
		}
	}
}
