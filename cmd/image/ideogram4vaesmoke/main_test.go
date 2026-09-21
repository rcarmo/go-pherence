package main

import (
	"encoding/binary"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSmokeRejectsGridBeforeOpeningWeights(t *testing.T) {
	for _, n := range []int{-1, 0, 65, int(^uint(0) >> 1)} {
		if err := runSmoke("missing", n, io.Discard); err == nil || !strings.Contains(err.Error(), "grid") {
			t.Fatal(n, err)
		}
	}
	if err := runSmoke("missing", 1, io.Discard); err == nil {
		t.Fatal("missing checkpoint accepted")
	}
}
func TestSmokeIncompleteCheckpointReturnsError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.safetensors")
	header := `{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	var size [8]byte
	binary.LittleEndian.PutUint64(size[:], uint64(len(header)))
	if err := os.WriteFile(path, append(append(size[:], header...), 0, 0, 0, 0), 0600); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 8; i++ {
		if err := runSmoke(path, 1, io.Discard); err == nil {
			t.Fatal("incomplete checkpoint accepted")
		}
	}
}
