package expertstream

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type fileRef struct {
	spec    DataFile
	absPath string
	file    *os.File
}

type expertLayout struct {
	spec    ExpertSpec
	file    *fileRef
	span    int64
	upRel   int64
	downRel int64
	spanEnd int64
}

type interval struct {
	start int64
	end   int64
	key   uint64
}

func openManifest(manifestPath string) (Manifest, map[string]*fileRef, map[uint64]*expertLayout, int64, error) {
	var manifest Manifest
	blob, err := os.ReadFile(manifestPath)
	if err != nil {
		return Manifest{}, nil, nil, 0, err
	}
	if err := json.Unmarshal(blob, &manifest); err != nil {
		return Manifest{}, nil, nil, 0, fmt.Errorf("%w: decode %s: %v", ErrInvalidManifest, manifestPath, err)
	}
	baseDir := filepath.Dir(manifestPath)
	files, err := validateAndOpenFiles(baseDir, manifest.Files)
	if err != nil {
		return Manifest{}, nil, nil, 0, err
	}
	layouts, maxSpan, err := validateManifest(manifest, files)
	if err != nil {
		closeFiles(files)
		return Manifest{}, nil, nil, 0, err
	}
	return manifest, files, layouts, maxSpan, nil
}

func validateAndOpenFiles(baseDir string, files []DataFile) (map[string]*fileRef, error) {
	if len(files) == 0 {
		return nil, fmt.Errorf("%w: no files", ErrInvalidManifest)
	}
	out := make(map[string]*fileRef, len(files))
	for _, spec := range files {
		if strings.TrimSpace(spec.ID) == "" {
			closeFiles(out)
			return nil, fmt.Errorf("%w: file id is empty", ErrInvalidManifest)
		}
		if _, ok := out[spec.ID]; ok {
			closeFiles(out)
			return nil, fmt.Errorf("%w: duplicate file id %q", ErrInvalidManifest, spec.ID)
		}
		if strings.TrimSpace(spec.Path) == "" {
			closeFiles(out)
			return nil, fmt.Errorf("%w: file %q path is empty", ErrInvalidManifest, spec.ID)
		}
		if spec.Size <= 0 {
			closeFiles(out)
			return nil, fmt.Errorf("%w: file %q size must be positive", ErrInvalidManifest, spec.ID)
		}
		wantHash, err := decodeSHA256(spec.SHA256)
		if err != nil {
			closeFiles(out)
			return nil, fmt.Errorf("%w: file %q sha256: %v", ErrInvalidManifest, spec.ID, err)
		}
		absPath := spec.Path
		if !filepath.IsAbs(absPath) {
			absPath = filepath.Join(baseDir, spec.Path)
		}
		f, err := os.Open(absPath)
		if err != nil {
			closeFiles(out)
			return nil, err
		}
		info, err := f.Stat()
		if err != nil {
			f.Close()
			closeFiles(out)
			return nil, err
		}
		if info.Size() != spec.Size {
			f.Close()
			closeFiles(out)
			return nil, fmt.Errorf("%w: file %q size mismatch got=%d want=%d", ErrChecksumMismatch, spec.ID, info.Size(), spec.Size)
		}
		haveHash, err := hashFileSHA256(f)
		if err != nil {
			f.Close()
			closeFiles(out)
			return nil, err
		}
		if !equalSHA256(haveHash, wantHash) {
			f.Close()
			closeFiles(out)
			return nil, fmt.Errorf("%w: file %q sha256 mismatch", ErrChecksumMismatch, spec.ID)
		}
		out[spec.ID] = &fileRef{spec: spec, absPath: absPath, file: f}
	}
	return out, nil
}

func validateManifest(manifest Manifest, files map[string]*fileRef) (map[uint64]*expertLayout, int64, error) {
	if manifest.Version != ManifestVersion {
		return nil, 0, fmt.Errorf("%w: version=%d want=%d", ErrInvalidManifest, manifest.Version, ManifestVersion)
	}
	if strings.TrimSpace(manifest.ModelID) == "" {
		return nil, 0, fmt.Errorf("%w: model_id is empty", ErrInvalidManifest)
	}
	if manifest.Layers <= 0 || manifest.Experts <= 0 {
		return nil, 0, fmt.Errorf("%w: layers=%d experts=%d must be positive", ErrInvalidManifest, manifest.Layers, manifest.Experts)
	}
	if !isPowerOfTwo(manifest.Alignment) {
		return nil, 0, fmt.Errorf("%w: alignment=%d must be a positive power of two", ErrInvalidManifest, manifest.Alignment)
	}
	wantEntries, err := checkedProduct(int64(manifest.Layers), int64(manifest.Experts))
	if err != nil {
		return nil, 0, fmt.Errorf("%w: layers*experts overflow", ErrInvalidManifest)
	}
	if int64(len(manifest.Entries)) != wantEntries {
		return nil, 0, fmt.Errorf("%w: entries=%d want=%d", ErrInvalidManifest, len(manifest.Entries), wantEntries)
	}
	layouts := make(map[uint64]*expertLayout, len(manifest.Entries))
	regions := make(map[string][]interval, len(files))
	var maxSpan int64
	for _, entry := range manifest.Entries {
		if _, ok := layouts[entry.Key]; ok {
			return nil, 0, fmt.Errorf("%w: duplicate expert key=%d", ErrInvalidManifest, entry.Key)
		}
		f := files[entry.File]
		if f == nil {
			return nil, 0, fmt.Errorf("%w: expert key=%d references unknown file %q", ErrInvalidManifest, entry.Key, entry.File)
		}
		if err := validateComponent(entry.Gate, "gate"); err != nil {
			return nil, 0, fmt.Errorf("%w: expert key=%d %v", ErrInvalidManifest, entry.Key, err)
		}
		if err := validateComponent(entry.Up, "up"); err != nil {
			return nil, 0, fmt.Errorf("%w: expert key=%d %v", ErrInvalidManifest, entry.Key, err)
		}
		if err := validateComponent(entry.Down, "down"); err != nil {
			return nil, 0, fmt.Errorf("%w: expert key=%d %v", ErrInvalidManifest, entry.Key, err)
		}
		if entry.Gate.Offset < 0 || entry.Gate.Offset%manifest.Alignment != 0 {
			return nil, 0, fmt.Errorf("%w: expert key=%d gate offset=%d is not aligned to %d", ErrInvalidManifest, entry.Key, entry.Gate.Offset, manifest.Alignment)
		}
		if entry.Up.Offset != entry.Gate.Offset+entry.Gate.Size {
			return nil, 0, fmt.Errorf("%w: expert key=%d up offset=%d is not contiguous after gate", ErrInvalidManifest, entry.Key, entry.Up.Offset)
		}
		if entry.Down.Offset != entry.Up.Offset+entry.Up.Size {
			return nil, 0, fmt.Errorf("%w: expert key=%d down offset=%d is not contiguous after up", ErrInvalidManifest, entry.Key, entry.Down.Offset)
		}
		spanEnd, err := checkedAdd(entry.Down.Offset, entry.Down.Size)
		if err != nil {
			return nil, 0, fmt.Errorf("%w: expert key=%d span overflow", ErrInvalidManifest, entry.Key)
		}
		if spanEnd > f.spec.Size {
			return nil, 0, fmt.Errorf("%w: expert key=%d span end=%d exceeds file %q size=%d", ErrInvalidManifest, entry.Key, spanEnd, entry.File, f.spec.Size)
		}
		span := spanEnd - entry.Gate.Offset
		if span <= 0 {
			return nil, 0, fmt.Errorf("%w: expert key=%d span must be positive", ErrInvalidManifest, entry.Key)
		}
		if span > maxSpan {
			maxSpan = span
		}
		layouts[entry.Key] = &expertLayout{
			spec:    cloneExpertSpec(entry),
			file:    f,
			span:    span,
			upRel:   entry.Up.Offset - entry.Gate.Offset,
			downRel: entry.Down.Offset - entry.Gate.Offset,
			spanEnd: spanEnd,
		}
		regions[entry.File] = append(regions[entry.File], interval{start: entry.Gate.Offset, end: spanEnd, key: entry.Key})
	}
	for fileID, spans := range regions {
		sort.Slice(spans, func(i, j int) bool {
			if spans[i].start != spans[j].start {
				return spans[i].start < spans[j].start
			}
			return spans[i].key < spans[j].key
		})
		for i := 1; i < len(spans); i++ {
			if spans[i].start < spans[i-1].end {
				return nil, 0, fmt.Errorf("%w: file %q experts %d and %d overlap", ErrInvalidManifest, fileID, spans[i-1].key, spans[i].key)
			}
		}
	}
	if maxSpan <= 0 {
		return nil, 0, fmt.Errorf("%w: no expert payloads", ErrInvalidManifest)
	}
	return layouts, maxSpan, nil
}

func validateComponent(spec ComponentSpec, name string) error {
	if spec.Offset < 0 {
		return fmt.Errorf("%s offset must be non-negative", name)
	}
	if spec.Size <= 0 {
		return fmt.Errorf("%s size must be positive", name)
	}
	dtype := normalizeDType(spec.DType)
	if dtype == "" {
		return fmt.Errorf("%s dtype is empty", name)
	}
	if len(spec.Shape) == 0 {
		return fmt.Errorf("%s shape is empty", name)
	}
	for _, dim := range spec.Shape {
		if dim <= 0 {
			return fmt.Errorf("%s shape contains non-positive dim %d", name, dim)
		}
	}
	if elemBytes, ok := dtypeBytes(dtype); ok {
		elements := int64(1)
		for _, dim := range spec.Shape {
			var err error
			elements, err = checkedProduct(elements, dim)
			if err != nil {
				return fmt.Errorf("%s shape overflows", name)
			}
		}
		wantBytes, err := checkedProduct(elements, int64(elemBytes))
		if err != nil {
			return fmt.Errorf("%s byte size overflows", name)
		}
		if spec.Size != wantBytes {
			return fmt.Errorf("%s size=%d does not match dtype=%s shape=%v bytes=%d", name, spec.Size, spec.DType, spec.Shape, wantBytes)
		}
	}
	return nil
}

func hashFileSHA256(f *os.File) ([]byte, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return nil, err
	}
	return h.Sum(nil), nil
}

func decodeSHA256(s string) ([]byte, error) {
	b, err := hex.DecodeString(strings.TrimSpace(s))
	if err != nil {
		return nil, err
	}
	if len(b) != sha256.Size {
		return nil, fmt.Errorf("want %d bytes got %d", sha256.Size, len(b))
	}
	return b, nil
}

func equalSHA256(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	var diff byte
	for i := range a {
		diff |= a[i] ^ b[i]
	}
	return diff == 0
}

func closeFiles(files map[string]*fileRef) {
	for _, f := range files {
		if f != nil && f.file != nil {
			_ = f.file.Close()
		}
	}
}

func cloneManifest(in Manifest) Manifest {
	out := in
	out.Files = append([]DataFile(nil), in.Files...)
	out.Entries = make([]ExpertSpec, len(in.Entries))
	for i, entry := range in.Entries {
		out.Entries[i] = cloneExpertSpec(entry)
	}
	return out
}

func cloneExpertSpec(in ExpertSpec) ExpertSpec {
	out := in
	out.Gate.Shape = append([]int64(nil), in.Gate.Shape...)
	out.Up.Shape = append([]int64(nil), in.Up.Shape...)
	out.Down.Shape = append([]int64(nil), in.Down.Shape...)
	return out
}
