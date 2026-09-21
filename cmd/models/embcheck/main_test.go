package main

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestRunBadArgumentsAndMalformedFile(t *testing.T) {
	var out, errs bytes.Buffer
	if run(nil, &out, &errs) == nil {
		t.Fatal("missing args")
	}
	path := filepath.Join(t.TempDir(), "bad.gguf")
	if err := os.WriteFile(path, []byte("NOPE"), 0o600); err != nil {
		t.Fatal(err)
	}
	if run([]string{path}, &out, &errs) == nil {
		t.Fatal("malformed file accepted")
	}
}
