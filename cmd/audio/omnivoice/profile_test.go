package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestProfilesNoOverwrite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "keep")
	if err := os.WriteFile(path, []byte("keep"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, pair := range [][2]string{{path, ""}, {"", path}, {path, path}} {
		if _, err := startProfiles(pair[0], pair[1]); err == nil {
			t.Fatal("existing profile overwritten")
		}
	}
	got, _ := os.ReadFile(path)
	if string(got) != "keep" {
		t.Fatal("changed file")
	}
}
func TestProfileFilesCreated(t *testing.T) {
	dir := t.TempDir()
	cpu, mem := filepath.Join(dir, "cpu"), filepath.Join(dir, "mem")
	stop, err := startProfiles(cpu, mem)
	if err != nil {
		t.Fatal(err)
	}
	stop()
	for _, name := range []string{cpu, mem} {
		stat, err := os.Stat(name)
		if err != nil || stat.Size() == 0 {
			t.Fatalf("profile %s missing", name)
		}
	}
}
