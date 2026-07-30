package expertstream

import (
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unsafe"
)

func TestOpen_ValidManifestAndAlignment(t *testing.T) {
	r, _, contents := mustOpenFixture(t, Options{Slots: 3, Workers: 2})
	defer r.Close()

	m := r.Manifest()
	if m.Version != ManifestVersion {
		t.Fatalf("version = %d, want %d", m.Version, ManifestVersion)
	}
	if m.Alignment != 64 {
		t.Fatalf("alignment = %d, want 64", m.Alignment)
	}
	if len(m.Entries) != 3 {
		t.Fatalf("entries = %d, want 3", len(m.Entries))
	}

	loaded, err := r.Load([]uint64{1, 2, 3})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for _, le := range loaded {
		layout := findEntry(t, m, le.Key)
		if le.Slot.Index%1 != 0 { // sanity: slot index always valid
		}
		if uintptrAligned(le.Slot.Bytes, 64) == false {
			t.Errorf("key=%d slot bytes not 64-byte aligned", le.Key)
		}
		wantGate := contents[layout.Gate.Offset : layout.Gate.Offset+layout.Gate.Size]
		if string(le.Gate.Bytes) != string(wantGate) {
			t.Errorf("key=%d gate bytes mismatch", le.Key)
		}
		wantUp := contents[layout.Up.Offset : layout.Up.Offset+layout.Up.Size]
		if string(le.Up.Bytes) != string(wantUp) {
			t.Errorf("key=%d up bytes mismatch", le.Key)
		}
		wantDown := contents[layout.Down.Offset : layout.Down.Offset+layout.Down.Size]
		if string(le.Down.Bytes) != string(wantDown) {
			t.Errorf("key=%d down bytes mismatch", le.Key)
		}
	}
}

func findEntry(t *testing.T, m Manifest, key uint64) ExpertSpec {
	t.Helper()
	for _, e := range m.Entries {
		if e.Key == key {
			return e
		}
	}
	t.Fatalf("entry key=%d not found", key)
	return ExpertSpec{}
}

func uintptrAligned(b []byte, alignment int64) bool {
	if len(b) == 0 {
		return true
	}
	addr := uintptr(unsafe.Pointer(&b[0]))
	return addr&(uintptr(alignment)-1) == 0
}

func TestOpen_SlotsMustBePositive(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	path := writeManifest(t, dir, manifest)
	if _, err := Open(path, Options{Slots: 0}); err == nil {
		t.Fatal("expected error for Slots=0")
	}
	if _, err := Open(path, Options{Slots: -1}); err == nil {
		t.Fatal("expected error for negative Slots")
	}
}

func TestOpen_WorkersDefaultsToOne(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	path := writeManifest(t, dir, manifest)

	r, err := Open(path, Options{Slots: 1, Workers: 0})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()
	if r.workers != 1 {
		t.Fatalf("workers = %d, want 1 for Workers=0", r.workers)
	}

	r2, err := Open(path, Options{Slots: 1, Workers: -5})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r2.Close()
	if r2.workers != 1 {
		t.Fatalf("workers = %d, want 1 for Workers=-5", r2.workers)
	}

	r3, err := Open(path, Options{Slots: 1, Workers: 7})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r3.Close()
	if r3.workers != 7 {
		t.Fatalf("workers = %d, want 7", r3.workers)
	}
}

func TestOpen_BadManifestJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "manifest.json")
	if err := os.WriteFile(path, []byte("{not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Open(path, Options{Slots: 1})
	if !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_MissingManifestFile(t *testing.T) {
	_, err := Open(filepath.Join(t.TempDir(), "nope.json"), Options{Slots: 1})
	if err == nil {
		t.Fatal("expected error for missing manifest file")
	}
}

func TestOpen_WrongVersion(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Version = 99
	path := writeManifest(t, dir, manifest)
	_, err := Open(path, Options{Slots: 1})
	if !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_EmptyModelID(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.ModelID = "   "
	path := writeManifest(t, dir, manifest)
	_, err := Open(path, Options{Slots: 1})
	if !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_NonPositiveLayersOrExperts(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Layers = 0
	path := writeManifest(t, dir, manifest)
	if _, err := Open(path, Options{Slots: 1}); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_AlignmentMustBePowerOfTwo(t *testing.T) {
	for _, alignment := range []int64{0, -8, 3, 6, 100} {
		dir := t.TempDir()
		manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
		manifest.Alignment = alignment
		path := writeManifest(t, dir, manifest)
		if _, err := Open(path, Options{Slots: 1}); !errors.Is(err, ErrInvalidManifest) {
			t.Fatalf("alignment=%d: err = %v, want ErrInvalidManifest", alignment, err)
		}
	}
}

func TestOpen_EntriesCountMismatch(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{
		{key: 1, gateN: 4, upN: 4, downN: 4},
		{key: 2, gateN: 4, upN: 4, downN: 4},
	})
	manifest.Layers = 1
	manifest.Experts = 5 // want 5 entries, only 2 present
	path := writeManifest(t, dir, manifest)
	if _, err := Open(path, Options{Slots: 2}); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_DuplicateExpertKey(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{
		{key: 1, gateN: 4, upN: 4, downN: 4},
		{key: 2, gateN: 4, upN: 4, downN: 4},
	})
	manifest.Entries[1].Key = 1 // duplicate of entries[0]
	manifest.Experts = 2
	path := writeManifest(t, dir, manifest)
	_, err := Open(path, Options{Slots: 2})
	if !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
	if !strings.Contains(err.Error(), "duplicate expert key") {
		t.Fatalf("err = %v, want duplicate expert key message", err)
	}
}

func TestOpen_UnknownFileReference(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Entries[0].File = "does-not-exist"
	path := writeManifest(t, dir, manifest)
	if _, err := Open(path, Options{Slots: 1}); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_MisalignedGateOffset(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Entries[0].Gate.Offset = 1 // not a multiple of 64
	path := writeManifest(t, dir, manifest)
	_, err := Open(path, Options{Slots: 1})
	if !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
	if !strings.Contains(err.Error(), "not aligned") {
		t.Fatalf("err = %v, want alignment message", err)
	}
}

func TestOpen_NonContiguousUpOffset(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Entries[0].Up.Offset += 4 // gap after gate
	path := writeManifest(t, dir, manifest)
	_, err := Open(path, Options{Slots: 1})
	if !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
	if !strings.Contains(err.Error(), "not contiguous after gate") {
		t.Fatalf("err = %v, want contiguity message", err)
	}
}

func TestOpen_NonContiguousDownOffset(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Entries[0].Down.Offset -= 4 // overlaps end of up
	path := writeManifest(t, dir, manifest)
	_, err := Open(path, Options{Slots: 1})
	if !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
	if !strings.Contains(err.Error(), "not contiguous after up") {
		t.Fatalf("err = %v, want contiguity message", err)
	}
}

func TestOpen_SpanExceedsFileSize(t *testing.T) {
	dir := t.TempDir()
	manifest, _, dataPath := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	// Shrink the recorded file size (and truncate the actual file) so the
	// expert span no longer fits, while keeping the hash of the new content.
	info, err := os.Stat(dataPath)
	if err != nil {
		t.Fatal(err)
	}
	shrink := info.Size() - 8
	if err := os.Truncate(dataPath, shrink); err != nil {
		t.Fatal(err)
	}
	newHash := hashFileForTest(t, dataPath)
	manifest.Files[0].Size = shrink
	manifest.Files[0].SHA256 = newHash
	path := writeManifest(t, dir, manifest)
	_, err = Open(path, Options{Slots: 1})
	if !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
	if !strings.Contains(err.Error(), "exceeds file") {
		t.Fatalf("err = %v, want span exceeds file message", err)
	}
}

func TestOpen_OverlappingExpertSpans(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{
		{key: 1, gateN: 4, upN: 4, downN: 4},
		{key: 2, gateN: 4, upN: 4, downN: 4},
	})
	// Force expert 2 to start inside expert 1's span.
	manifest.Entries[1].Gate.Offset = manifest.Entries[0].Gate.Offset
	manifest.Entries[1].Up.Offset = manifest.Entries[1].Gate.Offset + manifest.Entries[1].Gate.Size
	manifest.Entries[1].Down.Offset = manifest.Entries[1].Up.Offset + manifest.Entries[1].Up.Size
	path := writeManifest(t, dir, manifest)
	_, err := Open(path, Options{Slots: 2})
	if !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
	if !strings.Contains(err.Error(), "overlap") {
		t.Fatalf("err = %v, want overlap message", err)
	}
}

func TestOpen_DTypeShapeSizeMismatch(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Entries[0].Gate.Size = manifest.Entries[0].Gate.Size + 4 // no longer matches f32 * shape
	path := writeManifest(t, dir, manifest)
	_, err := Open(path, Options{Slots: 1})
	if !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_ShapeNonPositiveDim(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Entries[0].Gate.Shape = []int64{0}
	path := writeManifest(t, dir, manifest)
	if _, err := Open(path, Options{Slots: 1}); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_UnknownDTypePassesThroughSizeCheck(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	// Unknown dtypes skip the byte-size cross-check entirely.
	manifest.Entries[0].Gate.DType = "custom-quant"
	path := writeManifest(t, dir, manifest)
	r, err := Open(path, Options{Slots: 1})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer r.Close()
}

func TestOpen_DuplicateFileID(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Files = append(manifest.Files, manifest.Files[0])
	path := writeManifest(t, dir, manifest)
	if _, err := Open(path, Options{Slots: 1}); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_NoFiles(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Files = nil
	path := writeManifest(t, dir, manifest)
	if _, err := Open(path, Options{Slots: 1}); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_FileSizeMismatch(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Files[0].Size += 1
	path := writeManifest(t, dir, manifest)
	_, err := Open(path, Options{Slots: 1})
	if !errors.Is(err, ErrChecksumMismatch) {
		t.Fatalf("err = %v, want ErrChecksumMismatch", err)
	}
}

func TestOpen_CorruptSHA256(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	// Valid hex, valid length, but wrong digest for the actual file content.
	manifest.Files[0].SHA256 = strings.Repeat("ab", 32)
	path := writeManifest(t, dir, manifest)
	_, err := Open(path, Options{Slots: 1})
	if !errors.Is(err, ErrChecksumMismatch) {
		t.Fatalf("err = %v, want ErrChecksumMismatch", err)
	}
	if !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("err = %v, want sha256 mismatch message", err)
	}
}

func TestOpen_MalformedSHA256Hex(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Files[0].SHA256 = "not-hex!!"
	path := writeManifest(t, dir, manifest)
	if _, err := Open(path, Options{Slots: 1}); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_SHA256WrongLength(t *testing.T) {
	dir := t.TempDir()
	manifest, _, _ := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	manifest.Files[0].SHA256 = "abcd"
	path := writeManifest(t, dir, manifest)
	if _, err := Open(path, Options{Slots: 1}); !errors.Is(err, ErrInvalidManifest) {
		t.Fatalf("err = %v, want ErrInvalidManifest", err)
	}
}

func TestOpen_MissingDataFile(t *testing.T) {
	dir := t.TempDir()
	manifest, _, dataPath := buildFixture(t, dir, 64, []expertDef{{key: 1, gateN: 4, upN: 4, downN: 4}})
	if err := os.Remove(dataPath); err != nil {
		t.Fatal(err)
	}
	path := writeManifest(t, dir, manifest)
	if _, err := Open(path, Options{Slots: 1}); err == nil {
		t.Fatal("expected error for missing data file")
	}
}

func hashFileForTest(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	h, err := hashFileSHA256(f)
	if err != nil {
		t.Fatal(err)
	}
	return hex.EncodeToString(h)
}
