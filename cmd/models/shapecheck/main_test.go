package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunBadArgsAndShortTensor(t *testing.T) {
	var out bytes.Buffer
	if run(nil, &out) == nil {
		t.Fatal("missing args")
	}
	// Minimal GGUF with a one-element norm: first4 formatting must not panic.
	var f bytes.Buffer
	f.WriteString("GGUF")
	put := func(v any) {
		if err := binary.Write(&f, binary.LittleEndian, v); err != nil {
			t.Fatal(err)
		}
	}
	put(uint32(3))
	put(uint64(1))
	put(uint64(0))
	name := "output_norm.weight"
	put(uint64(len(name)))
	f.WriteString(name)
	put(uint32(1))
	put(uint64(1))
	put(uint32(0))
	put(uint64(0))
	for f.Len()%32 != 0 {
		f.WriteByte(0)
	}
	put(float32(1))
	path := filepath.Join(t.TempDir(), "small.gguf")
	if err := os.WriteFile(path, f.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{path}, &out); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "first=[1]") {
		t.Fatal(out.String())
	}
}
