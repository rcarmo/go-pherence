//go:build linux

package gguf

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime/debug"
	"testing"
)

func TestOpenClosesFileOnSemanticHeaderErrors(t *testing.T) {
	oldGC := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(oldGC)
	version := make([]byte, 8)
	copy(version, "GGUF")
	binary.LittleEndian.PutUint32(version[4:], 99)
	count := make([]byte, 16)
	copy(count, "GGUF")
	binary.LittleEndian.PutUint32(count[4:], 3)
	binary.LittleEndian.PutUint64(count[8:], ggufMaxCollectionCount+1)
	for name, data := range map[string][]byte{"magic": []byte("NOPE"), "version": version, "count": count} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "invalid.gguf")
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			resolved, err := filepath.EvalSymlinks(path)
			if err != nil {
				t.Fatal(err)
			}
			for i := 0; i < 8; i++ {
				if g, err := Open(path); err == nil {
					g.Close()
					t.Fatal("invalid header accepted")
				}
			}
			entries, err := os.ReadDir("/proc/self/fd")
			if err != nil {
				t.Skip(err)
			}
			for _, e := range entries {
				target, err := os.Readlink(filepath.Join("/proc/self/fd", e.Name()))
				if err == nil && target == resolved {
					t.Fatalf("GGUF descriptor %s leaked after rejected %s", e.Name(), name)
				}
			}
		})
	}
}
