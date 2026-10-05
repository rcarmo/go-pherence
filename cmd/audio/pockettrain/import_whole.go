//go:build linux

package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

func importWholeRequested(args []string) bool {
	for i, a := range args {
		if a == "-mode=import-whole" || a == "--mode=import-whole" || (a == "-mode" || a == "--mode") && i+1 < len(args) && args[i+1] == "import-whole" {
			return true
		}
	}
	return false
}
func importWholeMain(args []string, out io.Writer) error {
	fs := flag.NewFlagSet("pockettrain import-whole", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	mode := fs.String("mode", "import-whole", "import-whole")
	checkpoint := fs.String("source-checkpoint", "", "immutable dataset checkpoint directory")
	split := fs.String("split-dir", "", "verified exported train/dev manifest directory")
	dest := fs.String("manifest-output", "", "NEW whole-utterance native manifest")
	maxRows := fs.Int("max-rows", 1000, "source row ceiling")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *mode != "import-whole" || fs.NArg() != 0 || *checkpoint == "" || *split == "" || *dest == "" || *maxRows < 2 || *maxRows > 100000 {
		return fmt.Errorf("invalid import-whole options")
	}
	if err := requireDerivedOutput(*checkpoint, *dest); err != nil {
		return err
	}
	// Verify complete immutable manifests first, not just selected WAVs.
	sourceSums, err := verifyChecksumDirectory(*checkpoint)
	if err != nil {
		return err
	}
	splitSums, err := verifyChecksumDirectory(*split)
	if err != nil {
		return err
	}
	if sourceSums["samples.jsonl"] == "" || splitSums["export.json"] == "" || splitSums["train.jsonl"] == "" || splitSums["dev.jsonl"] == "" {
		return fmt.Errorf("required source/export manifests not checksummed")
	}
	source, sourcePin, err := boundedBytes(filepath.Join(*checkpoint, "samples.jsonl"), 8<<20)
	if err != nil {
		return err
	}
	export, exportPin, err := boundedBytes(filepath.Join(*split, "export.json"), 64<<10)
	if err != nil {
		return err
	}
	if sourcePin.SHA256 != sourceSums["samples.jsonl"] || exportPin.SHA256 != splitSums["export.json"] {
		return fmt.Errorf("source/export changed after checksum verification")
	}
	var declaration struct {
		Version    int    `json:"version"`
		Checkpoint string `json:"checkpoint"`
		Manifest   string `json:"source_manifest_sha256"`
	}
	if _, err := strictObject(export); err != nil {
		return err
	}
	if err := json.Unmarshal(export, &declaration); err != nil {
		return err
	}
	actual, err := filepath.EvalSymlinks(*checkpoint)
	if err != nil {
		return err
	}
	actual, err = filepath.Abs(actual)
	if err != nil {
		return err
	}
	declared, err := filepath.EvalSymlinks(declaration.Checkpoint)
	if err != nil {
		return err
	}
	declared, err = filepath.Abs(declared)
	if err != nil {
		return err
	}
	if declaration.Version != 1 || actual != declared || declaration.Manifest != sourcePin.SHA256 {
		return fmt.Errorf("export/source checkpoint identity mismatch")
	}
	train, trainPin, err := boundedBytes(filepath.Join(*split, "train.jsonl"), 8<<20)
	if err != nil {
		return err
	}
	dev, devPin, err := boundedBytes(filepath.Join(*split, "dev.jsonl"), 8<<20)
	if err != nil {
		return err
	}
	if trainPin.SHA256 != splitSums["train.jsonl"] || devPin.SHA256 != splitSums["dev.jsonl"] {
		return fmt.Errorf("splits changed after checksum verification")
	}
	rows, err := importWholeSelection(source, train, dev, actual, *maxRows)
	if err != nil {
		return err
	}
	root, err := os.OpenRoot(actual)
	if err != nil {
		return err
	}
	defer root.Close()
	if err := verifyWholeSourceTranscripts(root, source, sourceSums, *maxRows); err != nil {
		return err
	}
	samples := make([]int, len(rows))
	for i, row := range rows {
		n, pin, err := scanWholeRecording(root, row.Path, 60*24000)
		if err != nil {
			return err
		}
		if pin.SHA256 != row.SHA256 {
			return fmt.Errorf("selected waveform pin mismatch")
		}
		samples[i] = n
	}
	// Do not silently exclude overlong targets or crop references. The derived
	// source selection must be revised explicitly before codec preparation.
	if _, err := wholeReferences(samples); err != nil {
		return err
	}
	var data bytes.Buffer
	enc := json.NewEncoder(&data)
	for _, row := range rows {
		if err := enc.Encode(row); err != nil {
			return err
		}
	}
	if err := publishWholeManifest(*dest, data.Bytes()); err != nil {
		return err
	}
	pin := sha256.Sum256(data.Bytes())
	return json.NewEncoder(out).Encode(map[string]any{"mode": "import-whole", "rows": len(rows), "manifest": *dest, "manifest_sha256": hex.EncodeToString(pin[:]), "source_manifest_sha256": sourcePin.SHA256, "policy": preparedWholePolicy, "quality_qa": "still_required", "training_executed": false})
}
func verifyChecksumDirectory(directory string) (map[string]string, error) {
	data, _, err := boundedBytes(filepath.Join(directory, "SHA256SUMS"), 1<<20)
	if err != nil {
		return nil, err
	}
	root, err := os.OpenRoot(directory)
	if err != nil {
		return nil, err
	}
	defer root.Close()
	seen := map[string]string{}
	s := bufio.NewScanner(bytes.NewReader(data))
	s.Buffer(make([]byte, 4096), 8192)
	for s.Scan() {
		line := s.Text()
		if len(line) < 67 || line[64] != ' ' || (line[65] != ' ' && line[65] != '*') {
			return nil, fmt.Errorf("invalid checksum line")
		}
		digest, path := line[:64], line[66:]
		if !validDigest(digest) || !filepath.IsLocal(path) || filepath.Clean(path) != path || seen[path] != "" || len(seen) >= 4096 {
			return nil, fmt.Errorf("invalid/duplicate checksum path")
		}
		seen[path] = digest
		f, err := root.OpenFile(path, os.O_RDONLY|syscall.O_NONBLOCK, 0)
		if err != nil {
			return nil, err
		}
		st, err := f.Stat()
		if err != nil {
			f.Close()
			return nil, err
		}
		if !st.Mode().IsRegular() || st.Size() <= 0 || st.Size() > 128<<20 {
			f.Close()
			return nil, fmt.Errorf("checksum asset byte/type ceiling")
		}
		hash := sha256.New()
		n, readErr := io.Copy(hash, io.LimitReader(f, st.Size()+1))
		closeErr := f.Close()
		if readErr != nil {
			return nil, readErr
		}
		if closeErr != nil {
			return nil, closeErr
		}
		if n != st.Size() || hex.EncodeToString(hash.Sum(nil)) != digest {
			return nil, fmt.Errorf("checksum mismatch: %s", path)
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}
	if len(seen) == 0 {
		return nil, fmt.Errorf("empty checksums")
	}
	return seen, nil
}
func verifyWholeSourceTranscripts(root *os.Root, source []byte, sums map[string]string, maxRows int) error {
	scanner := bufio.NewScanner(bytes.NewReader(source))
	scanner.Buffer(make([]byte, 4096), 64<<10)
	count := 0
	for scanner.Scan() {
		count++
		if count > maxRows {
			return fmt.Errorf("source transcript row ceiling")
		}
		var row struct {
			Audio string `json:"audio"`
			File  string `json:"transcript_file"`
			Text  string `json:"transcript"`
			SHA   string `json:"source_sha256"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return err
		}
		if !filepath.IsLocal(row.File) || filepath.Clean(row.File) != row.File || sums[row.File] == "" || sums[row.Audio] != row.SHA {
			return fmt.Errorf("source transcript/audio not checksummed")
		}
		f, err := root.OpenFile(row.File, os.O_RDONLY|syscall.O_NONBLOCK, 0)
		if err != nil {
			return err
		}
		st, err := f.Stat()
		if err != nil {
			f.Close()
			return err
		}
		if !st.Mode().IsRegular() || st.Size() <= 0 || st.Size() > 8192 {
			f.Close()
			return fmt.Errorf("transcript file ceiling")
		}
		data, readErr := io.ReadAll(io.LimitReader(f, 8193))
		closeErr := f.Close()
		if readErr != nil {
			return readErr
		}
		if closeErr != nil {
			return closeErr
		}
		sum := sha256.Sum256(data)
		if len(data) > 8192 || hex.EncodeToString(sum[:]) != sums[row.File] || strings.TrimRight(string(data), "\r\n") != row.Text {
			return fmt.Errorf("transcript text/hash mismatch")
		}
	}
	return scanner.Err()
}

// Verify source pairing against both split lists before deriving train rows.
// Historical absolute paths must resolve to the admitted checkpoint-relative
// source path; manifest output contains only the portable source-relative path.
func importWholeSelection(source, train, dev []byte, checkpoint string, maxRows int) ([]wholeEntry, error) {
	type sourceRow struct {
		ID             string          `json:"id"`
		Audio          string          `json:"audio"`
		TranscriptFile string          `json:"transcript_file"`
		Transcript     string          `json:"transcript"`
		SHA            string          `json:"source_sha256"`
		Annotated      bool            `json:"quality_annotation_present"`
		Notes          json.RawMessage `json:"notes"`
	}
	sources := map[string]sourceRow{}
	scan := func(data []byte, visit func([]byte) error) error {
		s := bufio.NewScanner(bytes.NewReader(data))
		s.Buffer(make([]byte, 4096), 64<<10)
		count := 0
		for s.Scan() {
			count++
			if count > maxRows {
				return fmt.Errorf("split/source row ceiling")
			}
			if _, err := strictObject(s.Bytes()); err != nil {
				return err
			}
			if err := visit(s.Bytes()); err != nil {
				return err
			}
		}
		return s.Err()
	}
	if err := scan(source, func(line []byte) error {
		var row sourceRow
		if err := json.Unmarshal(line, &row); err != nil {
			return err
		}
		if row.ID == "" || sources[row.ID].ID != "" || !filepath.IsLocal(row.Audio) || filepath.Clean(row.Audio) != row.Audio || !validDigest(row.SHA) {
			return fmt.Errorf("invalid source row")
		}
		sources[row.ID] = row
		return nil
	}); err != nil {
		return nil, err
	}
	seenIDs, seenHashes := map[string]bool{}, map[string]bool{}
	var result []wholeEntry
	for index, data := range [][]byte{train, dev} {
		count := 0
		err := scan(data, func(line []byte) error {
			var row struct {
				ID       string `json:"id"`
				Audio    string `json:"audio_path"`
				Text     string `json:"text"`
				Language string `json:"language_id"`
			}
			dec := json.NewDecoder(bytes.NewReader(line))
			dec.DisallowUnknownFields()
			if err := dec.Decode(&row); err != nil {
				return err
			}
			s, ok := sources[row.ID]
			if !ok || seenIDs[row.ID] || seenHashes[s.SHA] || row.Text != s.Transcript || strings.TrimSpace(row.Text) == "" || row.Language != "en" || s.Annotated || (len(s.Notes) > 0 && string(s.Notes) != "null" && string(s.Notes) != `""`) {
				return fmt.Errorf("split/source mismatch, annotated row or leakage")
			}
			if !filepath.IsAbs(row.Audio) || filepath.Clean(row.Audio) != filepath.Join(checkpoint, s.Audio) {
				return fmt.Errorf("split audio path mismatch")
			}
			seenIDs[row.ID], seenHashes[s.SHA] = true, true
			count++
			if index == 0 {
				result = append(result, wholeEntry{ID: s.ID, Path: s.Audio, Transcript: s.Transcript, SHA256: s.SHA, Split: "train"})
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
		if count == 0 {
			return nil, fmt.Errorf("empty train/dev split")
		}
	}
	if len(result) < 2 {
		return nil, fmt.Errorf("need2training utterances")
	}
	data := bytes.Buffer{}
	for _, row := range result {
		json.NewEncoder(&data).Encode(row)
	}
	return readWholeManifest(data.Bytes(), maxRows)
}

// Resolve existing output parent as well as source root: a symlink must not
// redirect a derived output into an immutable checkpoint.
func requireDerivedOutput(checkpoint, path string) error {
	root, err := filepath.EvalSymlinks(checkpoint)
	if err != nil {
		return err
	}
	root, err = filepath.Abs(root)
	if err != nil {
		return err
	}
	parent, err := filepath.EvalSymlinks(filepath.Dir(path))
	if err != nil {
		return err
	}
	parent, err = filepath.Abs(parent)
	if err != nil {
		return err
	}
	relative, err := filepath.Rel(root, filepath.Join(parent, filepath.Base(path)))
	if err != nil {
		return err
	}
	if relative == "." || filepath.IsLocal(relative) {
		return fmt.Errorf("derived output must be outside immutable source checkpoint")
	}
	return nil
}
func publishWholeManifest(path string, data []byte) error {
	if _, err := os.Lstat(path); err == nil || !os.IsNotExist(err) {
		return fmt.Errorf("manifest destination must be new")
	}
	temp, err := os.CreateTemp(filepath.Dir(path), ".whole-manifest-*")
	if err != nil {
		return err
	}
	defer os.Remove(temp.Name())
	defer temp.Close()
	if _, err := temp.Write(data); err != nil {
		return err
	}
	if err := temp.Sync(); err != nil {
		return err
	}
	if err := temp.Close(); err != nil {
		return err
	}
	if err := publishNewExport(temp.Name(), path); err != nil {
		return err
	}
	return nil
}
