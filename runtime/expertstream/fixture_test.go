package expertstream

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
)

// expertDef describes one expert's component sizes (in f32 elements) used to
// build deterministic test fixtures.
type expertDef struct {
	key               uint64
	gateN, upN, downN int64
}

// buildFixture writes a single backing data file containing gate/up/down
// payloads for each expert (aligned + contiguous, in ascending key order) and
// returns a ready-to-serialize Manifest plus the raw file bytes for
// out-of-band assertions.
func buildFixture(t testing.TB, dir string, alignment int64, defs []expertDef) (Manifest, []byte, string) {
	t.Helper()
	rng := rand.New(rand.NewSource(42))
	entries := make([]ExpertSpec, 0, len(defs))
	var buf []byte
	var offset int64
	writeComponent := func(n int64) (int64, int64, []byte) {
		size := n * 4
		b := make([]byte, size)
		rng.Read(b)
		return offset, size, b
	}
	for _, d := range defs {
		aligned, err := alignUp(offset, alignment)
		if err != nil {
			t.Fatalf("alignUp: %v", err)
		}
		if aligned > int64(len(buf)) {
			buf = append(buf, make([]byte, aligned-int64(len(buf)))...)
		}
		offset = aligned

		gateOff, gateSize, gateBytes := writeComponent(d.gateN)
		buf = append(buf, gateBytes...)
		offset += gateSize

		upOff, upSize, upBytes := writeComponent(d.upN)
		buf = append(buf, upBytes...)
		offset += upSize

		downOff, downSize, downBytes := writeComponent(d.downN)
		buf = append(buf, downBytes...)
		offset += downSize

		entries = append(entries, ExpertSpec{
			Key:  d.key,
			File: "data0",
			Gate: ComponentSpec{Offset: gateOff, Size: gateSize, DType: "f32", Shape: []int64{d.gateN}},
			Up:   ComponentSpec{Offset: upOff, Size: upSize, DType: "f32", Shape: []int64{d.upN}},
			Down: ComponentSpec{Offset: downOff, Size: downSize, DType: "f32", Shape: []int64{d.downN}},
		})
	}

	dataPath := filepath.Join(dir, "data0.bin")
	if err := os.WriteFile(dataPath, buf, 0o644); err != nil {
		t.Fatalf("write data file: %v", err)
	}
	sum := sha256.Sum256(buf)

	manifest := Manifest{
		Version:   ManifestVersion,
		ModelID:   "test-model",
		Layers:    1,
		Experts:   len(defs),
		Alignment: alignment,
		Files: []DataFile{{
			ID:     "data0",
			Path:   "data0.bin",
			Size:   int64(len(buf)),
			SHA256: hex.EncodeToString(sum[:]),
		}},
		Entries: entries,
	}
	return manifest, buf, dataPath
}

// writeManifest serializes m as JSON into dir/manifest.json and returns its path.
func writeManifest(t testing.TB, dir string, m Manifest) string {
	t.Helper()
	blob, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		t.Fatalf("marshal manifest: %v", err)
	}
	path := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
	return path
}

// mustOpen builds+writes a default 3-expert fixture and opens a Reader.
func mustOpenFixture(t *testing.T, opts Options) (*Reader, string, []byte) {
	t.Helper()
	dir := t.TempDir()
	defs := []expertDef{
		{key: 1, gateN: 8, upN: 8, downN: 8},
		{key: 2, gateN: 16, upN: 16, downN: 16},
		{key: 3, gateN: 4, upN: 4, downN: 4},
	}
	manifest, contents, _ := buildFixture(t, dir, 64, defs)
	path := writeManifest(t, dir, manifest)
	r, err := Open(path, opts)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return r, path, contents
}
