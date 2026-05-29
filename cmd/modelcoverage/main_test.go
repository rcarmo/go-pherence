package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestSummarizeManifest(t *testing.T) {
	m := manifest{Version: 1, Families: map[string]manifestFamily{
		"b": {Status: "two", ValidationTarget: "make test", Coverage: map[string]bool{"done": true, "todo": false}},
		"a": {Status: "one", ValidationTarget: "make test", Coverage: map[string]bool{"done": true}},
	}}
	s, err := summarize(m, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(s) != 2 || s[0].Name != "a" || s[1].Name != "b" {
		t.Fatalf("summaries=%+v", s)
	}
	if s[1].Covered != 1 || s[1].Pending != 1 || len(s[1].PendingKeys) != 1 || s[1].PendingKeys[0] != "todo" {
		t.Fatalf("summary=%+v", s[1])
	}
	filtered, err := summarize(m, "b")
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered) != 1 || filtered[0].Name != "b" {
		t.Fatalf("filtered=%+v", filtered)
	}
	if _, err := summarize(m, "missing"); err == nil {
		t.Fatal("expected unknown family error")
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
