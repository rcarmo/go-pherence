//go:build linux

package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

func writeCheckedFixture(t *testing.T, dir string, files map[string][]byte) {
	t.Helper()
	var names []string
	for name, data := range files {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, data, 0600); err != nil {
			t.Fatal(err)
		}
		names = append(names, name)
	}
	sort.Strings(names)
	var sums strings.Builder
	for _, name := range names {
		sum := sha256.Sum256(files[name])
		fmt.Fprintf(&sums, "%x  %s\n", sum, name)
	}
	if err := os.WriteFile(filepath.Join(dir, "SHA256SUMS"), []byte(sums.String()), 0600); err != nil {
		t.Fatal(err)
	}
}
func importFixture(t *testing.T) (string, string) {
	t.Helper()
	root := t.TempDir()
	sourceDir, splitDir := filepath.Join(root, "checkpoint"), filepath.Join(root, "split")
	files := map[string][]byte{}
	var source, train, dev bytes.Buffer
	for i := 0; i < 3; i++ {
		id := fmt.Sprint(i)
		path := fmt.Sprintf("audio/%d.wav", i)
		text := fmt.Sprintf("transcripts/%d.txt", i)
		data := wavFixture((i+1)*1920, 1, 16, 1, 24000)
		sum := sha256.Sum256(data)
		files[path] = data
		files[text] = []byte(id + "\n")
		json.NewEncoder(&source).Encode(map[string]any{"id": id, "audio": path, "transcript_file": text, "transcript": id, "source_sha256": hex.EncodeToString(sum[:]), "notes": nil, "quality_annotation_present": false})
		writer := &train
		if i == 2 {
			writer = &dev
		}
		json.NewEncoder(writer).Encode(map[string]any{"id": id, "audio_path": filepath.Join(sourceDir, path), "text": id, "language_id": "en"})
	}
	files["samples.jsonl"] = source.Bytes()
	writeCheckedFixture(t, sourceDir, files)
	sum := sha256.Sum256(source.Bytes())
	export, _ := json.Marshal(map[string]any{"version": 1, "checkpoint": sourceDir, "source_manifest_sha256": hex.EncodeToString(sum[:]), "training_export_qa": "pending"})
	writeCheckedFixture(t, splitDir, map[string][]byte{"train.jsonl": train.Bytes(), "dev.jsonl": dev.Bytes(), "export.json": export})
	return sourceDir, splitDir
}
func TestPreparedWholeImportCLI(t *testing.T) {
	checkpoint, split := importFixture(t)
	dest := filepath.Join(t.TempDir(), "whole.jsonl")
	var out bytes.Buffer
	args := []string{"-mode=import-whole", "-source-checkpoint=" + checkpoint, "-split-dir=" + split, "-manifest-output=" + dest}
	if err := importWholeMain(args, &out); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(dest)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := readWholeManifest(data, 10)
	if err != nil || len(rows) != 2 || rows[0].Path != "audio/0.wav" {
		t.Fatal(rows, err)
	}
	if err := importWholeMain(args, &out); err == nil {
		t.Fatal("existing output replaced")
	}
	if !importWholeRequested([]string{"--mode", "import-whole"}) || importWholeRequested([]string{"-mode=train"}) {
		t.Fatal("dispatch")
	}
	if err := os.WriteFile(filepath.Join(checkpoint, "audio/0.wav"), []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	args[len(args)-1] = "-manifest-output=" + filepath.Join(t.TempDir(), "bad.jsonl")
	if err := importWholeMain(args, &out); err == nil {
		t.Fatal("changed source admitted")
	}
}
func TestPreparedWholeImportImmutableOutput(t *testing.T) {
	checkpoint, _ := importFixture(t)
	if err := requireDerivedOutput(checkpoint, filepath.Join(checkpoint, "derived.jsonl")); err == nil {
		t.Fatal("immutable output admitted")
	}
	outside := t.TempDir()
	link := filepath.Join(outside, "inside")
	if err := os.Symlink(checkpoint, link); err != nil {
		t.Fatal(err)
	}
	if err := requireDerivedOutput(checkpoint, filepath.Join(link, "derived.jsonl")); err == nil {
		t.Fatal("symlink output admitted")
	}
	if err := requireDerivedOutput(checkpoint, filepath.Join(outside, "derived.jsonl")); err != nil {
		t.Fatal(err)
	}
}
func TestPreparedWholeImportIdentityAndTranscriptRejection(t *testing.T) {
	for _, mutate := range []func(string, string){
		func(checkpoint, split string) {
			data := []byte(`{"version":1,"checkpoint":"/missing","source_manifest_sha256":"bad"}`)
			writeCheckedFixture(t, split, map[string][]byte{"export.json": data, "train.jsonl": []byte("{}\n"), "dev.jsonl": []byte("{}\n")})
		},
		func(checkpoint, split string) {
			// Rehashing the wrong transcript does not make it match approved text.
			p := filepath.Join(checkpoint, "transcripts/0.txt")
			data := []byte("wrong\n")
			if err := os.WriteFile(p, data, 0600); err != nil {
				t.Fatal(err)
			}
			sums, err := os.ReadFile(filepath.Join(checkpoint, "SHA256SUMS"))
			if err != nil {
				t.Fatal(err)
			}
			old := sha256.Sum256([]byte("0\n"))
			changed := sha256.Sum256(data)
			sums = bytes.Replace(sums, []byte(fmt.Sprintf("%x", old)), []byte(fmt.Sprintf("%x", changed)), 1)
			if err := os.WriteFile(filepath.Join(checkpoint, "SHA256SUMS"), sums, 0600); err != nil {
				t.Fatal(err)
			}
		},
	} {
		checkpoint, split := importFixture(t)
		mutate(checkpoint, split)
		dest := filepath.Join(t.TempDir(), "rejected.jsonl")
		if err := importWholeMain([]string{"-source-checkpoint=" + checkpoint, "-split-dir=" + split, "-manifest-output=" + dest}, &bytes.Buffer{}); err == nil {
			t.Fatal("invalid identity/transcript admitted")
		}
		if _, err := os.Stat(dest); !os.IsNotExist(err) {
			t.Fatal("rejection published manifest", err)
		}
	}
}
func TestPreparedWholeImportRejectsLeakage(t *testing.T) {
	checkpoint, split := importFixture(t)
	source, _ := os.ReadFile(filepath.Join(checkpoint, "samples.jsonl"))
	train, _ := os.ReadFile(filepath.Join(split, "train.jsonl"))
	dev, _ := os.ReadFile(filepath.Join(split, "dev.jsonl"))
	for _, tc := range []struct{ source, train, dev []byte }{{source, train, train}, {source, bytes.ReplaceAll(train, []byte(`"text":"0"`), []byte(`"text":"wrong"`)), dev}, {source, bytes.ReplaceAll(train, []byte(checkpoint), []byte("/elsewhere")), dev}, {source, train, nil}, {bytes.ReplaceAll(source, []byte(`"quality_annotation_present":false`), []byte(`"quality_annotation_present":true`)), train, dev}} {
		if _, err := importWholeSelection(tc.source, tc.train, tc.dev, checkpoint, 10); err == nil {
			t.Fatal("invalid selection admitted")
		}
	}
	if _, err := verifyChecksumDirectory(split); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(split, "SHA256SUMS"), []byte(strings.Repeat("a", 64)+"  ../escape\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := verifyChecksumDirectory(split); err == nil {
		t.Fatal("escaping checksum admitted")
	}
}
